def main():

    # VN-0 =                x
    # VN-1 =           x * (x @ B)
    # VN-2 =      x * (x * (x @ B)) @ B
    # VN-3 = x * (x * (x * (x @ B)) @ B) @ B
    #        1M   1M   1M   1M  MM    MM   MM

    def gen1():
        print("gen1 init")
        a = 0
        while True:
            print("gen1 before")
            next_val = yield
            print("gen1 after")
            if next_val is None:
                print("gen1 return")
                return a
            a += next_val
            print(f"a val {a}")
        return

    def gen2():
        print("gen2 init")
        b = 0
        while True:
            print("gen2 before")
            next_val = yield from gen1()
            print("gen2 after")
            if next_val is None:
                print("gen2 return")
                return b
            b += next_val
            print(f"b val {b}")
        return

    x = 5
    gen = gen2()
    print("Sending None")
    gen.send(None)
    print("Sending x")
    gen.send(x)
    print("Sending x")
    gen.send(x)
    print("Sending None")
    gen.send(None)
    print("Sending x")
    gen.send(x)
    print("Sending x")
    gen.send(x)
    # for i in range(3):
    #     print(gen.send(i))

    # def accumulate():
    #     tally = 0
    #     while 1:
    #         next_val = yield
    #         if next_val is None:
    #             return tally
    #         tally += next_val

    # def gather_tallies(tallies):
    #     while 1:
    #         tally = yield from accumulate()
    #         tallies.append(tally)

    # tallies = []
    # acc = gather_tallies(tallies)
    # acc.send(None)
    # for i in range(4):
    #     acc.send(i)

    # acc.send(None)

    # for i in range(5):
    #     acc.send(i)

    # acc.send(None)

    # print(tallies)


if __name__ == "__main__":
    main()
