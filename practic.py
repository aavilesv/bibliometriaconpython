name = False 
print(name)

def big_funcion(*args, **kwargs):
    global name
    name = True 
    print(name)
    print(args)
    print(kwargs)
    return args, kwargs

print(big_funcion(1,2,3, a=4, b=5))



def requiere_auth(func):
    def wrapper(user):
        if user.lower() == "admin":
            return func(user)
        else:
            return "Acceso denegado"
    return wrapper
@requiere_auth
def admin_dashboard(user):
    return f"Bienvenido al panel de administración, {user}"
print(admin_dashboard("admin"))  # Acceso permitido
print(admin_dashboard("guest"))  # Acceso denegado  

class BankAccount:
    def __init__(self, owner, initial_balance):
        self.owner = owner
        self.__balance = initial_balance  # Encapsulación

    def deposit(self, amount):
        if amount > 0:
            self.__balance += amount

    def withdraw(self, amount):
        if 0 < amount <= self.__balance:
            self.__balance -= amount
        else:
            print("Saldo insuficiente o monto inválido.")

    def check_balance(self):
        return f"Saldo actual: ${self.__balance}"


account = BankAccount("Ricardo", 1000)  # Abstracción

account.deposit(500)
# account.withdraw(700)

print(account.check_balance())
from abc import ABC, abstractmethod

class BankAccount(ABC):
    def __init__(self, owner, initial_balance):
        self.owner = owner
        self.__balance = initial_balance  # Encapsulación

    def deposit(self, amount):
        if amount > 0:
            self.__balance += amount

    def _get_balance(self):
        return self.__balance

    def _set_balance(self, new_balance):
        self.__balance = new_balance

    @abstractmethod
    def withdraw(self, amount):
        pass  # Polimorfismo

    def check_balance(self):
        return f"Saldo actual: ${self.__balance}"
