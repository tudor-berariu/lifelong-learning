from utils.elasticsearch_utils import get_server_reports

if __name__ == "__main__":

    d = get_server_reports(experiments=["3d_datasets_ewc_256_mlp"],
                           include_keys=["_eval_trace.[_].[_].[_].acc",
                                         "_args.train._optimizer.name.<type>"], smart_group=[0, 1])
    print(d)
