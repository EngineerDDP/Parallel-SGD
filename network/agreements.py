
class General:

    Type = 'Type'
    From = 'From'
    To = 'SendTo'


class Initialize:

    Type = 'Init'
    Init_Weight = 'Init_Weights'
    Node_ID = 'Node_ID'
    Return = 'Init_Back'
    Weight_Content = 'WContent'
    Redundancy = 'R'
    Nodes = 'N'
    Batch_Size = 'B'
    Current_State = 'Check'
    State_OK = 'OK'
    State_Hold = 'Hold'
    CodeType = 'CodeType'
    SyncClass = 'SyncState'
    Epoches = 'EPO'
    LOSS = 'LOSS_FUNCTION'
    Learn_Rate = 'LRate'


class Data:

    Type = 'Init_Data'
    Train_Data = 'TDATA'
    Eval_Data = 'EDATA'

class Transfer:

    Type = 'Transfer'
    ContentType = 'WType'


class DefaultNodes:

    Initialization_Server = -1
    Parameter_Server = -2