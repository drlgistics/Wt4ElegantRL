'''
Descripttion: Automatically generated file comment
version: 
Author: Wesley
Date: 2021-07-27 09:53:43
LastEditors: Wesley
LastEditTime: 2021-08-13 15:26:16
'''
from ctypes import cdll, c_char_p
from .PlatformHelper import PlatformHelper as ph
from wtpy.WtUtilDefs import singleton
import os

# Python对接C接口的库
@singleton
class WtDtWrapper:
    '''
    Wt平台数据组件C接口底层对接模块
    '''

    # api可以作为公共变量
    api = None
    ver = "Unknown"
    
    # 构造函数，传入动态库名
    def __init__(self):
        paths = os.path.split(__file__)
        dllname = ph.getModule("WtDtPorter")
        a = (paths[:-1] + (dllname,))
        _path = os.path.join(*a)
        self.api = cdll.LoadLibrary(_path)
        self.api.get_version.restype = c_char_p
        self.ver = bytes.decode(self.api.get_version())

    def run_datakit(self):
        '''
        启动数据组件
        '''
        self.api.start()

    def write_log(self, level, message:str, catName:str = ""):
        '''
        向组件输出日志
        '''
        self.api.write_log(level, bytes(message, encoding = "utf8").decode('utf-8').encode('gbk'), bytes(catName, encoding = "utf8"))

    def initialize(self, cfgfile:str = "dtcfg.json", logprofile:str = "logcfgdt.json"):
        '''
        C接口初始化
        '''
        try:
            self.api.initialize(bytes(cfgfile, encoding = "utf8"), bytes(logprofile, encoding = "utf8"))
        except OSError as oe:
            print(oe)

        self.write_log(102, "WonderTrader datakit initialzied，version：%s" % (self.ver))