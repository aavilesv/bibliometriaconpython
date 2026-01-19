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
