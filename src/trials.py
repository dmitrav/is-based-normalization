

import pandas

if __name__ == "__main__":

    b = {
        'name2': [],
        'name1': [],
        'name3': []
    }

    a = [
        [1,2,3],
        [4,5,6],
        [1,2,3],
        [4,5,6],
        [7,8,9]
    ]

    df = pandas.DataFrame(a, columns=b.keys())

    print(df)