import cv2, numpy as np, os, random
base = "sample_data"
os.makedirs(base, exist_ok=True)
for split in ['train','val']:
    for cls in ['defective','non_defective']:
        os.makedirs(os.path.join(base, split, cls), exist_ok=True)

def save(img, split, cls, idx):
    path = os.path.join(base, split, cls, f"{cls}_{idx}.png")
    cv2.imwrite(path, img)

def make_defective(h=80,w=80):
    img = np.ones((h,w,3), dtype=np.uint8)*255
    # random jagged line (simulated crack)
    pts = []
    x = random.randint(5,10)
    for _ in range(6):
        y = random.randint(5,w-5)
        pts.append((x,y))
        x += random.randint(8,12)
        if x>=w-5: break
    for i in range(len(pts)-1):
        cv2.line(img, pts[i], pts[i+1], (0,0,0), 2)
    return img

def make_nondefect(h=80,w=80):
    img = np.ones((h,w,3), dtype=np.uint8)*255
    # add small random spots
    for _ in range(3):
        cx = random.randint(10,w-10)
        cy = random.randint(10,h-10)
        r = random.randint(3,7)
        cv2.circle(img, (cx,cy), r, (0,0,0), -1)
    return img

for split in ['train','val']:
    n = 50 if split=='train' else 10
    for i in range(n):
        save(make_defective(), split, 'defective', i)
        save(make_nondefect(), split, 'non_defective', i)

print("Sample dataset created at ./sample_data/")
