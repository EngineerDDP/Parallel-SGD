if __name__ == '__main__':

    from network import NodeAssignment, Request
    from myExe import myExecutor

    nodes = NodeAssignment()
    nodes.add(0, '10.196.83.205')
    net = Request()

    from roles import Coordinator, Reclaimer
    with net.request(nodes) as req:
        master = Coordinator(req)
        master.submit_job(myExecutor)
        master.join()
        # master = Reclaimer(req)
        # master.require_client_log()

