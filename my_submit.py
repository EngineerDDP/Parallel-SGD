if __name__ == '__main__':

    # 导入节点分配容器
    from network import NodeAssignment
    # 导入请求处理
    from network import Request
    # 导入自定义执行类型，注意：自定义类型不能和下面的代码放在一起
    from myExe import myExecutor

    # 添加一个ip作为Worker
    nodes = NodeAssignment()
    nodes.add(0, '10.196.83.205')
    net = Request()

    # 增加协调者角色
    from roles import Coordinator

    # 发起一个请求
    # 如果客户端已经启动了，则可以直接提交，无需将代码更新至客户端。
    with net.request(nodes) as req:
        # 在请求的集群上创建一个协调者
        master = Coordinator(req)
        # 提交任务
        master.submit_job(myExecutor)
        # 等待执行完成
        master.join()

