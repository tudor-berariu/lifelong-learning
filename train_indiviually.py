import torch
import torch.nn as nn
from typing import Type, Callable

# Project imports
from my_types import Args, Tasks, Model, LongVector, DatasetTasks


def train_individually(model_class: Type,
                       get_optimizer: Callable[nn.Module],
                       tasks: Tasks,
                       args: Args)-> None:

    print(f"Training {clr('individually', attrs=['bold']):s} on all tasks.")

    model.train()
    model.use_softmax = False

    seen, total_epochs_no = 0, 0
    trace, best_results = [], None
    results = {}

    task_args = [(d_name, p_idx)
                 for d_name, p_no in zip(args.datasets, args.perms_no)
                 for p_idx in range(p_no)]
    order_tasks(task_args, args)

    for task_no, (dataset, p_idx) in enumerate(task_args):
        print(f"Training on task {task_no:d}: {dataset:s}-{(p_idx+1):03d}.")
        not_changed = 0
        crt_epochs = 0
        task = tasks[dataset]
        i_perm = task.perms[0][p_idx]
        t_perm = None if task.perms[1] is None else task.perms[1][p_idx]
        results = {}

        while crt_epochs < args.epochs_per_task:

            for data, target in task.train_loader:
                data, target = permute(data, target, i_perm, t_perm)

                optimizer.zero_grad()
                output = model(data)
                loss = F.cross_entropy(output, target)
                loss.backward()
                optimizer.step()

            seen += len(task.train_loader.dataset)
            crt_epochs += 1
            total_epochs_no += 1

            if total_epochs_no % args.eval_freq == 0:
                acc, loss = test(model, task.test_loader, i_perm, t_perm)
                results[dataset] = {"acc": acc, "loss": loss}

                model.train()
                model.use_softmax = False

                show_results(seen, results, best_results)
                best_results, changed = update_results(results, best_results)
                not_changed = 0 if changed else (not_changed + 1)
                if not changed:
                    print(f"No improvement for {not_changed:d} evals!!")
                trace.append((seen, total_epochs_no, results))
                e_df = results_to_dataframe(results)
                e_df['seen'] = seen
                e_df['epoch'] = total_epochs_no
                add_exp_params(e_df, args)
                if eval_df is None:
                    eval_df = e_df
                else:
                    eval_df = pd.concat([eval_df, e_df]).reset_index(drop=True)

                if len(trace) % args.save_freq == 0:
                    train_df = pd.DataFrame(train_info)
                    add_exp_params(train_df, args)

                    train_df.to_pickle(os.path.join(
                        args.out_dir,
                        f"epoch__{total_epochs_no:d}__losses.pkl"))
                    eval_df.to_pickle(os.path.join(
                        args.out_dir,
                        f"epoch__{total_epochs_no:d}__results.pkl"))
                    torch.save(trace, os.path.join(
                        args.out_dir,
                        f"epoch__{total_epochs_no:d}__trace.pkl"))
                    torch.save(elastic_info, os.path.join(
                        args.out_dir,
                        f"epoch__{total_epochs_no:d}__elastic_info.pkl"))
                if not_changed > 0 and args.stop_if_not_better == not_changed:
                    break

    train_df = pd.DataFrame(train_info)
    train_df['title'] = args.title
    add_exp_params(train_df, args)
    train_df.to_pickle(os.path.join(args.out_dir, f"losses.pkl"))
    eval_df.to_pickle(os.path.join(args.out_dir, f"results.pkl"))
    torch.save(trace, os.path.join(args.out_dir, f"trace.pkl"))
    torch.save(elastic_info, os.path.join(args.out_dir, f"elastic_info.pkl"))
