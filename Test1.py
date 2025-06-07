class BankAccount:
    def __init__(self,account_number,balance):
        self.account_number=account_number
        self.balance=balance
    def deposit(self,amount):
        self.balance+=amount
        print(self.balance)
    def withdraw(self,amount):
        self.balance-=amount
        print(self.balance)

a=BankAccount(123456,100)
a.deposit(10)
a.withdraw(20)
