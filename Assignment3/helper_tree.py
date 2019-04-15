"""
Hey you need to help?
Your code is getting messy.
Write in me, I'm a helper file.
(Alternate Universe where files can talk, lol, what more can happen!)
Absurdity is a God!
"""

def get_nodes_stats_depth_wise(node):
	"""hold my beer while I get the node stats depth"""

	depth = 0
	stats = {}
	num_nodes = 1
	childs = []
	childs.append(node)
	while len(childs)!=0:
		stats[depth] = num_nodes
		
		newchilds = []
		for n in childs:
			for child in n.children:
				newchilds.append(child)
				num_nodes += 1
		childs = newchilds

		depth += 1
	return stats

def breadth_first_search(node):
	"""HMB while I do breadth first search"""

	tmplist = []
	tmplist.append(node)
	num_nodes=1
	nodes_list = []

	while num_nodes != 0:
		tmp = tmplist.pop(0)
		num_nodes -= 1
		nodes_list.append(tmp)

		for child in tmp.children:
			tmplist.append(child)
			num_nodes += 1

	nodes_list.reverse()
	return nodes_list

def wait():
	while True:
		pass
