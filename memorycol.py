# https://stackoverflow.com/questions/552744/how-do-i-profile-memory-usage-in-python 
from collections import Counter
import linecache
import os
import tracemalloc
""" ==== fast use ====
import memorycol as mc
mc.tracemalloc.start()

snapshot = mc.tracemalloc.take_snapshot()
mc.display_top(snapshot)
"""
# ==============================================================
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
# print(f"{bcolors.HEADER}Whatsup\n{bcolors.OKBLUE}Whatsup\n{bcolors.OKCYAN}Whatsup\n{bcolors.OKGREEN}Whatsup\n{bcolors.WARNING}Whatsup\n{bcolors.FAIL}Whatsup\n{bcolors.ENDC}Whatsup\n{bcolors.BOLD}Whatsup\n{bcolors.UNDERLINE}Whatsup{bcolors.ENDC}\n")
# ==============================================================

def display_top(snapshot, key_type='lineno', limit=3):
	snapshot = snapshot.filter_traces((
		tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
		tracemalloc.Filter(False, "<unknown>"),
	))
	top_stats = snapshot.statistics(key_type)
	print(f"============ Memory Usage ============\nTop {limit} lines")
	for index, stat in enumerate(top_stats[:limit], 1):
		frame = stat.traceback[0]
		# replace "/path/to/module/file.py" with "module/file.py"
		filename = os.sep.join(frame.filename.split(os.sep)[-2:])
		# printing info
		print(f"#{index}: {filename}:{bcolors.FAIL}{frame.lineno}{bcolors.ENDC}: {bcolors.WARNING}{(stat.size / 1024):.1f} KiB{bcolors.ENDC}")
		line = linecache.getline(frame.filename, frame.lineno).strip()
		if line:
			print('    %s' % line)
	other = top_stats[limit:]
	if other:
		size = sum(stat.size for stat in other)
		print("%s other: %.1f KiB" % (len(other), size / 1024))
	total = sum(stat.size for stat in top_stats)
	print(f"Total allocated size: {bcolors.WARNING}{(total / 1024):.1f} KiB{bcolors.ENDC}")


# ===================== example usage ===================== 
# # start
# tracemalloc.start()
# 
# # launch task
# counts = Counter()
# fname = '/usr/share/dict/american-english'
# with open(fname) as words:
#     words = list(words)
#     for word in words:
#         prefix = word[:3]
#         counts[prefix] += 1
# print('Top prefixes:', counts.most_common(3))
# 
# # Analyse
# snapshot = tracemalloc.take_snapshot()
# display_top(snapshot)

 