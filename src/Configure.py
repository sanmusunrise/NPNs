
class Configure:
    def __init__(self,conf_file = "config.cfg"):
        self.confs = {}
        for line in open(conf_file):
            line = line.strip().split()
            if len(line) <3:
                continue
            key,value,type = line
            self.confs[key] = eval(type +"('" + value + "')" )
    def __setitem__(self, key, value):
        self.confs[key] = value
    def __getitem__(self, key):
        return self.confs[key]