from user import User
from card import Card
import random


class ATM:
    def __init__(self, alluser):
        self.alluser = alluser

    def randomiCardId(self):  # 随机生成开户卡号
        while True:
            str_data = ''  # 存储卡号
            for i in range(6):  # 随机生成6位卡号
                ch = chr(random.randrange(ord('0'), ord('9') + 1))
                str_data += ch
            if not self.alluser.get(str):  # 判断卡号是否重复
                return str_data

    def creatUser(self):
        # 目标向用户字典中添加一个键值对（卡号、用户对象）
        name = input("请输入姓名:")
        Uid = input("请输入身份证号:")
        phone = input("请输入手机号:")
        prestMoney = float(input("请输入预存金额:"))
        if prestMoney <= 0:
            print("预存款输入有误，开户失败")
            return -1
        oncePwd = input("请输入密码：")
        passWord = input("请再次输入密码：")
        if passWord != oncePwd:
            print("两次密码输入不同......")
            return -1
        print("密码设置成功，请牢记密码:  %s " % passWord)
        cardId = self.randomiCardId()
        card = Card(cardId, oncePwd, prestMoney)  # 创建卡
        user = User(name, Uid, phone, card)  # 创建用户
        self.alluser[cardId] = user  # 存入用户字典
        print("您的开户以完成，请牢记开户账号: %s" % cardId)

    def checkpwg(self, realPwd):
        for i in range(3):
            psd2 = input("请输入密码：")
            if realPwd == psd2:
                return True
        print("密码输错三次，系统自动退出......")
        return False

    def lockCard(self):
        inptcardId = input("请输入您的卡号：")
        user = self.alluser.get(inptcardId)
        if not self.alluser.get(inptcardId):
            print("此卡号不存在...锁定失败！")
            return -1
        if user.card.cardLock:
            print("该卡已经被锁定，无需再次锁定！")
            return -1
        if not self.checkpwg(user.card.cardPwd):  # 验证密码
            print("密码错误...锁定失败！！")
            return -1
        user.card.cardLock = True
        print("该卡被锁定成功！")

    def searchUser(self, base=1):
        if base == 2:
            inptcardId = input("请输入转出主卡号：")
        elif base == 3:
            inptcardId = input("请转入卡号：")
        elif base == 1:
            inptcardId = input("请输您的卡号：")
        user = self.alluser.get(inptcardId)
        # 如果卡号不存在，下面代码就会执行
        if not self.alluser.get(inptcardId):
            print("此卡号不存在...查询失败！")
            return -1
        if user.card.cardLock:
            print("该用户已经被锁定...请解卡后使用！")
            return -1
        if not self.checkpwg(user.card.cardPwd):  # 验证密码
            print("密码错误过多...卡已经被锁定，请解卡后使用！")
            user.card.cardLock = True
            return -1

        if not base == 3:  # 查询转入账户   不打印余额
            print("账户: %s   余额: %.2f  " %
                  (user.card.cardId, user.card.money))
        return user

    def getMoney(self):
        userTF = self.searchUser()
        if userTF != -1:
            if userTF.card.cardId != '':
                inptMoney = float(input("请输入取款金额:"))
                if inptMoney > int(userTF.card.money):
                    print("输入的金额大于余额，请先查询余额！")
                    return -1
                userTF.card.money = \
                    float(userTF.card.money) - inptMoney
                print("取款成功！  账户: %s   余额: %.2f  " %
                      (userTF.card.cardId, userTF.card.money))
        else:
            return -1

    def saveMoney(self):
        userTF = self.searchUser()
        if userTF != -1:
            if not userTF.card.cardLock == True:
                if userTF.card.cardId != '':
                    inptMoney = float(input("请输入要存入得金额:"))
                    if inptMoney < 0:
                        print("请输入正确金额")
                    else:
                        userTF.card.money += inptMoney
                        print("存款成功！  账户: %s   余额: %.2f  " %
                              (userTF.card.cardId, userTF.card.money))
        else:
            return -1

    def transferMoney(self):
        MasterTF = self.searchUser(base=2)
        if (MasterTF == -1):
            return -1
        userTF = self.searchUser(base=3)
        if (userTF == -1):
            return -1
        in_tr_Money = float(input("请输入转账金额："))
        if MasterTF.card.money >= in_tr_Money:
            str = input("您确认要继续转账操作吗（y/n）？：")
            if str.upper() == "Y":
                MasterTF.card.money -= in_tr_Money
                userTF.card.money += in_tr_Money
                print("转账成功！  账户: %s   余额: %.2f  " %
                      (MasterTF.card.cardId, MasterTF.card.money))
            else:
                print("转账失败，中止了操作")
        else:
            print("转账失败,余额不足！  账户: %s   余额: %.2f  " %
                  (MasterTF.card.cardId, MasterTF.card.money))

    def unlockCard(self):
        inptcardId = input("请输入您的卡号：")
        user = self.alluser.get(inptcardId)
        while 1:
            if not self.alluser.get(inptcardId):
                print("此卡号不存在...解锁失败！")
                return -1
            elif not user.card.cardLock:
                print("该卡未被锁定，无需解锁！")
                break
            elif not self.checkpwg(user.card.cardPwd):
                print("密码错误...解锁失败！！")
                return -1
                user.card.cardLock = False  # 解锁

            print("该卡 解锁 成功！")
            break
