
class Node(object):
	def __init__(self,x,y,value):
		self.x = x
		self.y = y
		self.value = value
	def printInfo(self):
		print(self.x,self.y,self.value)


def AtmosphericLight(darkChannel, img):
    height = darkChannel.shape[0]
    width = darkChannel.shape[1]
    nodes = []
    # 用一个链表结构(list)存储数据
    for i in range(0, height):
        for j in range(0, width):
            oneNode = Node(i, j, darkChannel[i, j])
            nodes.append(oneNode)
    # 排序
    nodes = sorted(nodes, key=lambda node: node.value, reverse=True)

    atomsphericLight  = img[nodes[0].x, nodes[0].y, :]
    return atomsphericLight

