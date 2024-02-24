FRAME_COUNTER = 0
PREDICTION_SCORES = []
OBJECTNESS_SCORES = []
FPS = []

file_path = f"cam2_rear/logs.txt"
f = open(file_path, 'r')
lines = f.readlines()
f.close()

def is_block_start(line):
    return line.startswith("2024-")

def parse_block(block):
    fps = block[-1].split("=")[1]
    FPS.append(float(fps))
    for i in range(len(block)):
        if block[i].startswith("objectness score"):
            pred_score = [block[j] for j in range(1, i)]
            obj_score = [block[k] for k in range(i, len(block)-1)]
            pred_score = "".join(pred_score)[len("pred_score=tensor("):-1]
            pred_score = pred_score[1:-1][1:-1]
            pred_score = pred_score.replace("]", " ").replace("[", " ").replace(",", " ")
            pred_score = pred_score.split(" ")
            for p in pred_score:
                if len(p) > 1:
                    try:
                        PREDICTION_SCORES.append(float(p))
                    except Exception as e:
                        continue
            obj_score = "".join(obj_score)
            obj_score = obj_score[len("objectness score=tensor("):obj_score.find("])")+1]
            obj_score = obj_score.replace("]", " ").replace("[", " ").replace(",", " ")
            obj_score = obj_score.split(" ")
            for o in obj_score:
                if len(o) > 1:
                    try:
                        OBJECTNESS_SCORES.append(float(o))
                    except Exception as e:
                        continue


i =0
blocks = []
while i<len(lines):
    if is_block_start(lines[i]):
        FRAME_COUNTER += 1
        i+=1
        block = []
        while i<len(lines) and len(lines[i])>4:
            block.append(lines[i].rstrip())
            blocks.append(block)
            i+=1
        blocks.append(block)
    i+=1


for block in blocks:
    parse_block(block)


FRAME_COUNTER += len(blocks)


def analyze_lst(lst):
    print(f"max     = {max(lst)}")
    print(f"min     = {min(lst)}")
    print(f"avg     = {sum(lst)/len(lst)}")
    print(f"median  = {sorted(lst)[len(lst)//2]}")


print("FRAME_COUNTER:")
print(FRAME_COUNTER)
print("")

print("PREDICTION_SCORES:")
analyze_lst(PREDICTION_SCORES)
print("")

print("OBJECTNESS_SCORES:")
analyze_lst(OBJECTNESS_SCORES)
print("")

print("FPS:")
analyze_lst(FPS)
print("")

exit(0)

print("PREDICTION_SCORES:")
print(PREDICTION_SCORES)
print("")

print("OBJECTNESS_SCORES:")
print(OBJECTNESS_SCORES)
print("")

print("FPS:")
print(FPS)
print("")
