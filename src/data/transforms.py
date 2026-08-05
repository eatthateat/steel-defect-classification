import cv2
from torchvision import transforms


def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    # v = np.median(gray)
    # sigma = 0.33
    # lower = int(max(0, (1.0 - sigma) * v))
    # upper = int(min(255, (1.0 + sigma) * v))
    # edges = cv2.Canny(enhanced, lower, upper)
    edges_3 = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    
    return edges_3


train_transform = transforms.Compose([
    transforms.Lambda(lambda x: preprocess_image(x)),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.Resize((200, 200)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(45),
    transforms.RandomVerticalFlip()
])

eval_transform = transforms.Compose([
    transforms.Lambda(lambda x: preprocess_image(x)),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.Resize((200, 200))
])

