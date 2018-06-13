from utils.elasticsearch_utils import get_server_reports

if __name__ == "__main__":
    full_report = get_server_reports(experiments=["3d_datasets_ewc_256_mlp"],
                           include_keys=["_eval_trace.[seen].[task_idx].[_].acc",
                                         "_eval_trace.[seen].[task_idx].[_].loss.<type>",
                                         "_args.train._optimizer.__name.<type>"], smart_group=1,
                                     df_format=True)

    print(full_report)
