1. transforms.Resize((200, 200)) - NEU датасет и так 200x200, а ResNet ждет 224x224.
2. class_to_idx молча возвращает None для неизвестных классов.
3. transforms.Normalize закоментированны в обоих траснформах.
4. imread в dataset молча вернет None для битого пути, нужно выкидывать исключение
5. проблема с параметрами в "image = cv2.imread(self.image_paths[idx], cv2.COLOR_BGR2RGB)", конверсии на самом деле не происходит.