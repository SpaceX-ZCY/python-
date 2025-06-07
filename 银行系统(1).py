from admin import Admin_bank
from atm import ATM
import time


class HomePage:
    def __init__(self):
        self.allUserD = {}  # 使用字典存储数据
        self.atm = ATM(self.allUserD)
        self.admin = Admin_bank()  # 管理员开机界面

    def saveUser(self):
        self.allUserD.update(self.atm.alluser)
        print("数据存盘成功")

    def main(self):
        self.admin.printAdminView()
        resL = self.admin.adminOption()
        if not resL:
            while True:
                self.admin.printsysFunctionView()
                option = input("请输入您的操作：")
                if option not in ("1", "2", "3", "4", "5",
                                  "6", "7", "S", "Q", "q"):
                    print("输入操作项有误,请仔细确认！")
                    time.sleep(1)
                if option == "1":  # 开户
                    self.atm.creatUser()
                elif option == "2":  # 查询
                    self.atm.searchUser()
                elif option == "3":  # 取款
                    self.atm.getMoney()
                elif option == "4":  # 存储
                    self.atm.saveMoney()
                elif option == "5":  # 转账
                    self.atm.transferMoney()
                elif option == "6":  # 锁定
                    self.atm.lockCard()
                elif option == "7":  # 解锁
                    self.atm.unlockCard()
                elif option.upper() == "Q":
                    if not (self.admin.adminOption()):
                        self.saveUser()
                        print('退出系统')
                        return -1


if __name__ == "__main__":
    homepage = HomePage()
    homepage.main()
