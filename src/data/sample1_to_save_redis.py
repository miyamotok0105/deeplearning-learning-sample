# -*- coding:utf-8 -*-

#redis start
#redis-server /usr/local/etc/redis.conf &
#
#redis-cli shutdown

import re
import time
import redis

class Snowflake:
    def __init__(self, machine_id, epoch=0, init_serial_no=0):
        self._machine_id = machine_id
        self._epoch = epoch
        self._serial_no = init_serial_no

    def generate(self):
        unique_id = (
            ((int(time.time() * 1000) - self._epoch) & 0x1ffffffffff) << 22 |
            (self._machine_id & 0x3ff) << 12 |
            (self._serial_no & 0xfff)
            )
        self._serial_no += 1
        return unique_id

class Stack:
    def __init__(self, stack = None):
        if type(stack) is type([]):
            self.stack = stack
        elif stack is None:
            self.stack = []
        else:
            print('\nError Message:\n\tYour input is not type of list!\n\tPlease enter list type!\n')

    def push(self, e):
        self.stack.append(e)    
        return self.stack       

    def pop(self):
        try:
            sEl = self.stack.pop()
            return sEl
        except IndexError:            
            print('\nError Message:\n\tThere is no element in the stack!\n')

def get_Snowflake(num):
    stk = Stack()
    snowflake = Snowflake(0, epoch=int(time.time() * 1000))
    for i in range(num):
        stk.push(snowflake.generate())
    return stk


r = redis.Redis(host='localhost', port=6379, db=0)
r.flushall()

index_count = 0
for line in open('sample1.txt', 'r'):
	index_count = index_count + 1

stk = get_Snowflake(index_count)
print("num of index is ",len(stk.stack))

for line in open('sample1.txt', 'r'):
	m = re.match("__label__[0-9]", line)
	# print(m.group(0)) #label
	# print(re.sub("__label__[0-9]", "", line)) #text
	r.hmset(stk.pop(), {'label': m.group(0), 'text': re.sub("__label__[0-9]", "", line)})

for key in r.scan_iter():
	print(r.hvals(key)[0])
	print(r.hvals(key)[1].decode('utf-8'))


