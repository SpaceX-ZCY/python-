class Card:
    def __init__(self,cardId,cardPwd,money):
        self.cardId = cardId
        self.cardPwd = cardPwd
        self.money = money
        self.cardLock = False
