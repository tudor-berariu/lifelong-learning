from utils.elasticsearch_utils import get_server_reports

if __name__ == "__main__":

    d = get_server_reports(["F1s272MBm5wd3rDHf_es"])
    print(d)
