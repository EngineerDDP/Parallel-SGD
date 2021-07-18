import executor


if __name__ == '__main__':
    executor.Cohort(save_trace_log=True).slave_forever()
