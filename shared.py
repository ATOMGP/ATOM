class Shared(object):

    def __init__(self):
        self.msg = ''
        self.progress = 0
    
    def read(self):
        ret = (self.msg, self.progress) 
        return ret
            
    def write(self, msg, progress):
        self.msg = msg
        self.progress = progress