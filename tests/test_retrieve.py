import executor.psgd as PSGD
import roles
import network


if __name__ == '__main__':

    nodes = PSGD.parse_worker(worker_cnt=9, ps=True)

    req = network.Request()

    with req.request(nodes) as com:
        roles.Reclaimer(com).require_client_log()
