class abc():
    def __init__(self) -> None:
        return
    
    def do_something(self):
        self.__do_some_static('used static method!')
        return

    @staticmethod
    def __do_some_static(arg):
        print(arg)
        return

if __name__ == "__main__":    
    obj = abc()
    obj.do_something()
    