import torch
from torch import Tensor
from typing import Union, List, Dict, Tuple, NamedTuple
import copy

# Project imports
from .ewc import EWC

Constraint = NamedTuple("Constraint", [("task_idx", int),
                                       ("epoch", int),
                                       ("mode", Dict[str, Tensor]),
                                       ("elasticity", Dict[str, Tensor])])


class TestConstraintImportance(EWC):
    def __init__(self, *args, **kwargs):
        super(TestConstraintImportance, self).__init__(*args, **kwargs)

        args = self._args

        agent_args = args.lifelong
        no_noise_scales = agent_args.no_noise_scales
        no_constraint_scales = agent_args.no_constraint_scales

        # Force constraint only on first task
        self.first_task_only = True
        # self.scale = agent_args.scale

        # Build Scales ( % )
        noise_scales = torch.linspace(0, 1, no_noise_scales)
        constraint_scales = torch.linspace(0, 1, no_constraint_scales)
        self.variations = [(a.item(), b.item()) for a in noise_scales for b in constraint_scales]

        print(f"(NOISE, SCALE):\n{self.variations}\n\n")

        # Necessary variables to restore training after first task
        self.first_task_constraint = None
        self.first_task_model = None
        self.first_task_optim = None
        self.first_eval_metrics = None

        self.constraint_max_norm = None # Constraint max possible value for normalization

    def train_sequentially(self):

        self._start_experiment()  # TEMPLATE

        # Load general Classes
        report = self.report
        multitask = self.multitask
        seen = 0

        # ------------------------------------------------------------------------------------------
        # -- TestConstraintImportance
        variations = self.variations
        max_task_idx = self.no_tasks - 1
        train_task_list = list(enumerate(self.train_tasks))
        train_task_list += train_task_list[1:] * (len(variations))
        c_noise = None
        c_scale = None
        done_variations = 0
        extra_info = None
        # ------------------------------------------------------------------------------------------

        for self.crt_task_idx, self.crt_data_loaders in train_task_list:
            early_stop_task = False
            eval_no_improvement = 0

            train_task_idx, data_loaders = self.crt_task_idx, self.crt_data_loaders

            train_loader, val_loader = data_loaders
            task_name = self.crt_task_name = train_loader.name
            self.crt_eval_epoch = 0

            print(f"Training on task {train_task_idx:d}: {task_name:s}.")

            self._start_train_task()  # TEMPLATE

            # Train each task for a number of epochs
            for self.crt_task_epoch in range(self.epochs_per_task):
                crt_epoch = self.crt_task_epoch

                # Train task 1 epoch
                train_loss, train_acc, seen, info = self._train_epoch(train_task_idx, train_loader,
                                                                      crt_epoch)

                # Get information to feed to reporting agent
                train_info = {"acc": train_acc, "loss": train_loss}
                train_info.update(info)
                report.trace_train(self.seen, train_task_idx, crt_epoch,
                                   self.all_epochs, train_info)

                # Evaluate
                if crt_epoch % self.eval_freq == 0 or crt_epoch == (self.epochs_per_task - 1):
                    eval_epoch = self.crt_eval_epoch

                    # Validate on first 'how_many' tasks
                    how_many = self.no_tasks if self.eval_not_trained else train_task_idx + 1
                    new_best_acc_cnt = new_best_loss_cnt = 0
                    for eval_task_idx, val_loader in enumerate(multitask.test_tasks(how_many)):
                        self.crt_eval_task_idx = eval_task_idx
                        val_loss, val_acc, info = self._eval_task(eval_task_idx, val_loader,
                                                                  crt_epoch, eval_epoch)

                        #  -- Reporting
                        val_info = {"acc": val_acc, "loss": val_loss}
                        val_info.update(info)

                        new_best_acc, new_best_loss = report.trace_eval(self.seen, eval_task_idx,
                                                                        crt_epoch, eval_epoch,
                                                                        self.all_eval_ind_epochs,
                                                                        val_info)
                        new_best_acc_cnt += new_best_acc
                        new_best_loss_cnt += new_best_loss

                        # Early task stop
                        if eval_task_idx == train_task_idx:
                            if new_best_acc + new_best_loss > 0:
                                eval_no_improvement = 0
                            else:
                                eval_no_improvement += 1
                                if eval_no_improvement > self.early_stop:
                                    early_stop_task = True

                        self.all_eval_ind_epochs += 1

                    self.crt_eval_epoch += 1
                    self.all_eval_epochs += 1
                    self.scheduler.step()

                if crt_epoch % self.save_report_freq == 0:
                    report.save()

                if self.stop_training or early_stop_task:
                    print(f"[TRAIN] Early stop task. stop_training: {self.stop_training}"
                          f" early_stop_task: {early_stop_task}")
                    break

                self.all_epochs += 1

            self._end_train_task()

            # --------------------------------------------------------------------------------------
            # -- TestConstraintImportance

            if self.crt_task_idx == 0:
                self.first_task_constraint = copy.deepcopy(self.constraints[0])
                self.first_task_model = copy.deepcopy(self._model)
                self.first_task_optim = copy.deepcopy(self._optimizer)
            elif self.crt_task_idx == max_task_idx and done_variations < len(self.variations):
                c_noise, c_scale = self.variations[done_variations]
                print('-' * 80)
                print(f"Done training last task. Restoring now with variation. "
                      f"Noise: {c_noise}; Scale: {c_scale}")
                print('-' * 80)

                # Adjust new constraint
                self.constraints = [self.modify_constraint(self.first_task_constraint,
                                                           c_scale, c_noise)]

                # Restore useful variables
                self._model = copy.deepcopy(self.first_task_model)
                self._optimizer = self.get_optimizer(self._model.parameters())
                self._optimizer.load_state_dict(self.first_task_optim.state_dict())

                done_variations += 1

            # --------------------------------------------------------------------------------------

            report.save()
            report.finished_training_task(train_task_idx+1, self.seen, info=extra_info)

            # --------------------------------------------------------------------------------------
            # -- TestConstraintImportance

            extra_info = None

            if self.crt_task_idx == max_task_idx:
                # Just finished reporting on a full training. - Copy eval Metrics for scores :)
                extra_info = {
                    "noise": c_noise,
                    "scale": c_scale,
                    "eval_metrics": copy.deepcopy(report._eval_metrics)
                }
                report._eval_metrics = copy.deepcopy(self.first_eval_metrics)
            elif self.crt_task_idx == 0:
                self.first_eval_metrics = copy.deepcopy(report._eval_metrics)

            # --------------------------------------------------------------------------------------

            if self.stop_training:
                break

        self._end_experiment()  # TEMPLATE

        report.save(final=True)

    def modify_constraint(self, constraint: Constraint, scale: float, noise: float):
        max_norm = self.constraint_max_norm
        elasticity = copy.deepcopy(constraint.elasticity)

        if max_norm is None:
            max_norm = 0
            for name, param in elasticity.items():
                max_norm = max(param.abs().max(), max_norm)
            self.constraint_max_norm = max_norm.item()

        if scale != 0:
            # Scale constraint with scale and add % noise of scaled max_norm
            scaled_max_norm = max_norm * scale * noise
            for name, param in elasticity.items():
                param.mul_(scale)
                noise_param = torch.zeros_like(param).uniform_(-scaled_max_norm, scaled_max_norm)
                param.add_(noise_param)

                min_param = param.min()
                if min_param < 0:
                    param.add_(min_param)

                param.div_(param.max()).mul_(scaled_max_norm)

        new_constraint = Constraint(constraint.task_idx, constraint.epoch,
                                    copy.deepcopy(constraint.mode), elasticity)

        return new_constraint



