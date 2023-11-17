from typing import Any, Optional

import torch
import torch.distributed as dist
import wandb
from coati.models.loss import DpoLoss
from coati.models.utils import calc_masked_log_probs
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from tqdm import trange
from transformers import PreTrainedTokenizerBase

from colossalai.booster import Booster
from colossalai.cluster import DistCoordinator
from colossalai.utils import get_current_device

from .base import SLTrainer
from .utils import is_rank_0, to_device


def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    dist.all_reduce(tensor=tensor, op=dist.ReduceOp.SUM)
    tensor.div_(dist.get_world_size())
    return tensor


class DPOTrainer(SLTrainer):
    """
        Trainer for PPO algorithm.

    Args:
        strategy (Strategy): the strategy to use for training
        actor (Actor): the actor model in ppo algorithm
        critic (Critic): the critic model in ppo algorithm
        reward_model (RewardModel): the reward model in rlhf algorithm to make reward of sentences
        initial_model (Actor): the initial model in rlhf algorithm to generate reference logics to limit the update of actor
        actor_optim (Optimizer): the optimizer to use for actor model
        critic_optim (Optimizer): the optimizer to use for critic model
        kl_coef (float, defaults to 0.1): the coefficient of kl divergence loss
        train_batch_size (int, defaults to 8): the batch size to use for training
        buffer_limit (int, defaults to 0): the max_size limitation of buffer
        buffer_cpu_offload (bool, defaults to True): whether to offload buffer to cpu
        eps_clip (float, defaults to 0.2): the clip coefficient of policy loss
        vf_coef (float, defaults to 1.0): the coefficient of value loss
        ptx_coef (float, defaults to 0.9): the coefficient of ptx loss
        value_clip (float, defaults to 0.4): the clip coefficient of value loss
        sample_buffer (bool, defaults to False): whether to sample from buffer
        dataloader_pin_memory (bool, defaults to True): whether to pin memory for data loader
        offload_inference_models (bool, defaults to True): whether to offload inference models to cpu during training process
        callbacks (List[Callback], defaults to []): the callbacks to call during training process
        generate_kwargs (dict, optional): the kwargs to use while model generating
    """

    def __init__(
        self,
        actor: Any,
        ref_model: Any,
        booster: Booster,
        actor_optim: Optimizer,
        actor_lr_scheduler: _LRScheduler,
        tokenizer: PreTrainedTokenizerBase,
        max_epochs: int = 1,
        beta: float = 0.1,
        accumulation_steps: int = 1,
        start_epoch: int = 0,
        save_interval: int = 0,
        save_dir: str = None,
        display_inference_result_interval=20,
        coordinator: DistCoordinator = None,
    ) -> None:
        super().__init__(booster, max_epochs=max_epochs, model=actor, optimizer=actor_optim, start_epoch=start_epoch)
        self.ref_model = ref_model
        self.actor_scheduler = actor_lr_scheduler
        self.tokenizer = tokenizer
        self.actor_loss_fn = DpoLoss(beta)
        self.save_interval = save_interval
        self.display_inference_result_interval = display_inference_result_interval
        self.coordinator = coordinator
        self.save_dir = save_dir
        self.num_train_step = 0
        self.accumulation_steps = accumulation_steps
        self.device = get_current_device()

    def _before_fit(
        self,
        train_preference_dataloader: DataLoader = None,
        eval_preference_dataloader: DataLoader = None,
        log_dir: Optional[str] = None,
        use_wandb: bool = False,
    ):
        """
        Args:
            prompt_dataloader (DataLoader): the dataloader to use for prompt data
            pretrain_dataloader (DataLoader): the dataloader to use for pretrain data
        """
        self.train_dataloader = train_preference_dataloader
        self.eval_dataloader = eval_preference_dataloader
        self.writer = None
        if use_wandb and is_rank_0():
            assert log_dir is not None, "log_dir must be provided when use_wandb is True"
            import wandb

            self.wandb_run = wandb.init(project="Coati-dpo", sync_tensorboard=True)
        if log_dir is not None and is_rank_0():
            import os
            import time

            from torch.utils.tensorboard import SummaryWriter

            log_dir = os.path.join(log_dir, "ppo")
            log_dir = os.path.join(log_dir, time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()))
            self.writer = SummaryWriter(log_dir=log_dir)

    def _train(self, epoch: int):
        """
        Args:
            epoch int: the number of current epoch
        """
        self.model.train()
        step_bar = trange(
            len(self.train_dataloader) // self.accumulation_steps,
            desc=f"Epoch {epoch + 1}/{self.max_epochs}",
            disable=not is_rank_0(),
        )
        for i, batch in enumerate(self.train_dataloader):
            batch = to_device(batch, self.device)
            # print(batch)
            (
                chosen_input_ids,
                chosen_attention_mask,
                chosen_loss_mask,
                reject_input_ids,
                reject_attention_mask,
                reject_loss_mask,
            ) = (
                batch["chosen_input_ids"],
                batch["chosen_attention_mask"],
                batch["chosen_loss_mask"],
                batch["reject_input_ids"],
                batch["reject_attention_mask"],
                batch["reject_loss_mask"],
            )
            torch.set_printoptions(threshold=20_000)
            # if is_rank_0():
            #     print(chosen_input_ids[0])
            #     print(chosen_attention_mask[0])
            #     print(chosen_loss_mask[0])
            #     print(reject_input_ids[0])
            #     print(reject_attention_mask[0])
            #     print(reject_loss_mask[0])
            #     masked_chosen = []
            #     masked_reject = []
            #     for j in range(chosen_input_ids[0].size(0)):
            #         if chosen_loss_mask[0][j]:
            #             masked_chosen.append(chosen_input_ids[0][j])
            #         else:
            #             masked_chosen.append(102)
            #         if reject_loss_mask[0][j]:
            #             masked_reject.append(reject_input_ids[0][j])
            #         else:
            #             masked_reject.append(102)
            # exit()

            batch_size = chosen_input_ids.size()[0]

            actor_all_logits = self.model(
                torch.cat([chosen_input_ids, reject_input_ids]),
                torch.cat([chosen_attention_mask, reject_attention_mask]),
            )["logits"].to(torch.float32)
            actor_chosen_logits = actor_all_logits[:batch_size]
            actor_reject_logits = actor_all_logits[batch_size:]

            logprob_actor_chosen = calc_masked_log_probs(actor_chosen_logits, chosen_input_ids, chosen_loss_mask[:, 1:])

            logprob_actor_reject = calc_masked_log_probs(actor_reject_logits, reject_input_ids, reject_loss_mask[:, 1:])

            self.ref_model.eval()
            with torch.no_grad():
                ref_all_logits = self.ref_model(
                    torch.cat([chosen_input_ids, reject_input_ids]),
                    torch.cat([chosen_attention_mask, reject_attention_mask]),
                )["logits"].to(torch.float32)
                ref_chosen_logits = ref_all_logits[:batch_size]
                ref_reject_logits = ref_all_logits[batch_size:]
                logprob_ref_chosen = calc_masked_log_probs(ref_chosen_logits, chosen_input_ids, chosen_loss_mask[:, 1:])
                logprob_ref_reject = calc_masked_log_probs(ref_reject_logits, reject_input_ids, reject_loss_mask[:, 1:])

            losses, chosen_rewards, rejected_rewards = self.actor_loss_fn(
                logprob_actor_chosen,
                logprob_actor_reject,
                logprob_ref_chosen if logprob_ref_chosen is not None else None,
                logprob_ref_reject if logprob_ref_reject is not None else None,
                chosen_loss_mask,
                reject_loss_mask,
            )
            reward_accuracies = (chosen_rewards > rejected_rewards).float()

            loss = losses.mean()

            self.booster.backward(loss=loss, optimizer=self.optimizer)
            if self.num_train_step % self.accumulation_steps == self.accumulation_steps - 1:
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.actor_scheduler.step()

            # if i % self.display_inference_result_interval == 0:
            #     self.tokenizer.padding_side = "left"
            #     self.ref_model.eval()
            #     with torch.no_grad():
            #         prompt_input_ids = chosen_input_ids.detach().clone()
            #         test_prompt = []
            #         for idx in range(batch_size):
            #             start = 0
            #             end = -1
            #             for j in range(prompt_input_ids.size(1)):
            #                 if prompt_input_ids[idx, j]!=self.tokenizer.pad_token_id and start==0:
            #                     start = j
            #                 if chosen_loss_mask[idx][j]==True:
            #                     end = j
            #                     test_prompt.append(self.tokenizer.decode(prompt_input_ids[idx, start:end], skip_special_tokens=False))
            #                     break
            #         # print(test_prompt)
            #         assert all([isinstance(s,str) for s in test_prompt])
            #         test_input_ids = self.tokenizer(test_prompt, return_tensors='pt', add_special_tokens=False, padding='max_length',
            #                                                max_length=1000, truncation=True).to(get_current_device())['input_ids']
            #         # if is_rank_0():
            #         #     print(test_input_ids[0])
            #         responses = generate(self.ref_model, test_input_ids,
            #             self.tokenizer, test_input_ids.size(1)+200, do_sample=True, temperature=0.9)
            #         for res in self.tokenizer.batch_decode(responses, skip_special_tokens=False):
            #             # TODO: I don't know why but using coordinator here trigger a group sync error. Need to fix it.
            #             if is_rank_0():
            #                 print("Reponse: ",res)
            #         # extra_padding = chosen_input_ids.size(1) - padded_prompt.size(1)
            #         # padded_prompt = torch.nn.functional.pad(padded_prompt, (extra_padding, 0), value=self.tokenizer.pad_token_id)
            #         # padded_atten_mask = torch.nn.functional.pad(padded_atten_mask, (extra_padding, 0), value=False)
            #         # self.coordinator.print_on_master(padded_prompt[0])
            #         # self.coordinator.print_on_master(padded_atten_mask[0])
            #         # res = generate(self.model, padded_prompt, self.tokenizer, padded_prompt.size(1)+200,
            #         #                 do_sample=True, temperature=0.9, attention_mask=padded_atten_mask,
            #         #                 prepare_inputs_fn = lambda x, **kwargs: {'input_ids':x, 'attention_mask':kwargs.get('attention_mask')})

            #         # self.coordinator.print_on_master(res[0])
            #         # debug_tokenzier = []
            #         # for pos in range(padded_prompt.size(1)):
            #         #     debug_tokenzier.append((self.tokenizer.decode(padded_prompt[0][pos].cpu()),padded_prompt[0][pos].cpu().item()))
            #         # self.coordinator.print_on_master(debug_tokenzier)
            #         # self.coordinator.print_on_master('test case:\n'.join(self.tokenizer.batch_decode(res, skip_special_tokens=True)))
            #     self.tokenizer.padding_side = "right"

            # sync
            loss_mean = all_reduce_mean(tensor=loss)
            chosen_rewards_mean = all_reduce_mean(tensor=chosen_rewards)
            rejected_rewards_mean = all_reduce_mean(tensor=rejected_rewards)
            reward_accuracies_mean = all_reduce_mean(tensor=reward_accuracies)

            # logging
            if self.writer and is_rank_0():
                self.writer.add_scalar("train/loss", loss_mean.to(torch.float16), self.num_train_step)
                self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]["lr"], self.num_train_step)
                self.writer.add_scalar(
                    "train/chosen_rewards", chosen_rewards_mean.mean().to(torch.float16), self.num_train_step
                )
                self.writer.add_scalar(
                    "train/rejected_rewards",
                    rejected_rewards_mean.mean().to(torch.float16),
                    self.num_train_step,
                )
                self.writer.add_scalar(
                    "train/accuracy",
                    reward_accuracies_mean.mean().to(torch.float16),
                    self.num_train_step,
                )

            if i % self.accumulation_steps == self.accumulation_steps - 1:
                self.num_train_step += 1
                step_bar.update()
        step_bar.close()

    def _eval(self, epoch: int):
        """
        Args:
            epoch int: the number of current epoch
        """
        if self.eval_dataloader is None:
            return
        self.model.eval()
        step_bar = trange(
            len(self.eval_dataloader),
            desc=f"Epoch {epoch + 1}/{self.max_epochs}",
            disable=not is_rank_0(),
        )
        eval_chosen_reward = []
        eval_rejected_reward = []
        eval_loss = []
        eval_accuracy = []
        responses = []

        with torch.no_grad():
            for i, batch in enumerate(self.eval_dataloader):
                batch = to_device(batch, self.device)
                (
                    chosen_input_ids,
                    chosen_attention_mask,
                    chosen_loss_mask,
                    reject_input_ids,
                    reject_attention_mask,
                    reject_loss_mask,
                ) = (
                    batch["chosen_input_ids"],
                    batch["chosen_attention_mask"],
                    batch["chosen_loss_mask"],
                    batch["reject_input_ids"],
                    batch["reject_attention_mask"],
                    batch["reject_loss_mask"],
                )

                batch_size = chosen_input_ids.size()[0]

                actor_all_logits = self.model(
                    torch.cat([chosen_input_ids, reject_input_ids]),
                    torch.cat([chosen_attention_mask, reject_attention_mask]),
                )["logits"].to(torch.float32)
                actor_chosen_logits = actor_all_logits[:batch_size]
                actor_reject_logits = actor_all_logits[batch_size:]

                logprob_actor_chosen = calc_masked_log_probs(
                    actor_chosen_logits, chosen_input_ids, chosen_loss_mask[:, 1:]
                )

                logprob_actor_reject = calc_masked_log_probs(
                    actor_reject_logits, reject_input_ids, reject_loss_mask[:, 1:]
                )

                self.ref_model.eval()

                ref_all_logits = self.ref_model(
                    torch.cat([chosen_input_ids, reject_input_ids]),
                    torch.cat([chosen_attention_mask, reject_attention_mask]),
                )["logits"].to(torch.float32)
                ref_chosen_logits = ref_all_logits[:batch_size]
                ref_reject_logits = ref_all_logits[batch_size:]
                logprob_ref_chosen = calc_masked_log_probs(ref_chosen_logits, chosen_input_ids, chosen_loss_mask[:, 1:])
                logprob_ref_reject = calc_masked_log_probs(ref_reject_logits, reject_input_ids, reject_loss_mask[:, 1:])

                losses, chosen_rewards, rejected_rewards = self.actor_loss_fn(
                    logprob_actor_chosen,
                    logprob_actor_reject,
                    logprob_ref_chosen if logprob_ref_chosen is not None else None,
                    logprob_ref_reject if logprob_ref_reject is not None else None,
                    chosen_loss_mask,
                    reject_loss_mask,
                )
                reward_accuracies = (chosen_rewards > rejected_rewards).float()

                if i == 0 and is_rank_0():
                    # sequences = generate(self.model.module, chosen_input_ids[:3,:20], self.tokenizer, **{'do_sample':True, 'max_length':100})
                    prompt_input_ids = chosen_input_ids.detach().clone()
                    for idx in range(batch_size):
                        for j in range(prompt_input_ids.size(1)):
                            if chosen_loss_mask[idx][j] == True:
                                prompt_input_ids[idx][j:] = self.tokenizer.pad_token_id
                                break
                    prompt_attention_mask = prompt_input_ids != self.tokenizer.pad_token_id
                    if isinstance(self.model.module, torch.nn.parallel.DistributedDataParallel):
                        res = self.model.module.module.generate(
                            prompt_input_ids,
                            attention_mask=prompt_attention_mask,
                            do_sample=True,
                            max_length=500,
                            temperature=0.75,
                            num_return_sequences=1,
                        )
                    else:
                        res = self.model.module.generate(
                            prompt_input_ids,
                            attention_mask=prompt_attention_mask,
                            do_sample=True,
                            max_length=500,
                            temperature=0.75,
                            num_return_sequences=1,
                        )

                    answer = self.tokenizer.batch_decode(res, skip_special_tokens=True)
                    prompt_eval = self.tokenizer.batch_decode(prompt_input_ids, skip_special_tokens=True)
                    for idx in range(batch_size):
                        responses.append(f"prompt:{prompt_eval[idx]}\noutput:{answer[idx]}")
                        self.coordinator.print_on_master(
                            f"###### test case {idx} #########\nprompt:{prompt_eval[idx]}\noutput:{answer[idx]}"
                        )

                loss = losses.mean()
                eval_chosen_reward.append(chosen_rewards.to(torch.float16).mean().item())
                eval_rejected_reward.append(rejected_rewards.to(torch.float16).mean().item())
                eval_loss.append(loss.to(torch.float16).item())
                eval_accuracy.append(reward_accuracies.to(torch.float16).mean().item())
                step_bar.update()
        if self.writer and is_rank_0():
            my_table = wandb.Table(
                columns=[f"sample response {idx}" for idx in range(len(responses))], data=[responses]
            )
            try:
                self.wandb_run.log({"sample_response": my_table})
            except OSError as e:
                print(e)
            self.writer.add_scalar("eval/loss", sum(eval_loss) / len(eval_loss), epoch)
            self.writer.add_scalar("eval/chosen_rewards", sum(eval_chosen_reward) / len(eval_chosen_reward), epoch)
            self.writer.add_scalar(
                "eval/rejected_rewards",
                sum(eval_rejected_reward) / len(eval_rejected_reward),
                epoch,
            )
            self.writer.add_scalar(
                "eval/accuracy",
                sum(eval_accuracy) / len(eval_accuracy),
                epoch,
            )
        step_bar.close()
