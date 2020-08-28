if __name__ == '__main__':

    from network import NodeAssignment, Request
    from myExe import myExecutor

    nodes = NodeAssignment()
    nodes.add(0, '192.168.1.136')
    net = Request()

    from roles import Coordinator
    with net.request(nodes) as req:
        master = Coordinator(req)
        master.submit_job(myExecutor)
        master.join()

