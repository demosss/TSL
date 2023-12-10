import torch

class FGSMTS():
    def __init__(self, model, loss_fn, device):
        """
        Методика состязательной атаки методом FSGM на временной ряд. 

        параметр: model - модель машинного обучения

        параметр: loss_fn - функция ошибки
        
        парамерт: device - вычислительное устройство (cpu/cuda)
        """
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.device = device




    def attack(self, X, y, eps=0.05, clip_min=None, clip_max=None, targeted = False, num_elem = None):
        """
        Функция реализующая атаку

        параметр: X - образец (образцы) временного ряда подлежащий(ие) атаке
        
        параметр: y - ожидаемый (истинный) результат работы модели

        параметр: eps - величина внесенных возмущений в исходный временной ряд

        параметр: clip_min - минимальное значение атакованного экземпляра временного ряда 

        параметр: clip_max - максимальное значение атакованного экземпляра временного ряда

        параметр: targeted - вид атаки целевая (targeted = True), нецелевая (targeted = False)

        параметр: num_elem - количество значимых коэффициентов автокорреляции временного ряда 
        (определяет количество последних элементов временного ряда, подлежащих атаке, None - атакуются все элементы)
        """

        # Проверка корректности выбранной величины eps
        if eps < 0:
            raise ValueError(
                "eps должен быть больше или равен 0, eps {} не допустим".format(eps)
            )   
        
        # Проверка корректности введеных значений clip_min и clip_max
        if clip_min is not None and clip_max is not None:
            if clip_min > clip_max:
                raise ValueError(
                    "clip_min должен быть меньше или равен clip_max, clip_min={}  clip_max={}".format(
                        clip_min, clip_max
                    )
                )

        # Перенос тензоров X и y на вычислительное устройство
        X = X.clone().detach().to(self.device)
        y = y.clone().detach().to(self.device)
        
        # Вычисление матрицы градиента
        self.model.train()
        X.requires_grad = True
        self.model.zero_grad()
        output = self.model(X)
        loss = self.loss_fn(output, y)

        # Инверсия ошибки для целевой атаки
        if targeted:
            loss = -loss

        grad = torch.autograd.grad(loss, X, retain_graph=False, create_graph=False)[0]


        # Обнуление градиентов элементов временного ряда не подлежащих атаке
        if num_elem is not None:
            grad[:, :-num_elem, :] = 0

        # Создание атакованного экземпляра временного ряда
        adv_X = X + eps * grad.sign()


        if (clip_min is not None) or (clip_max is not None):
            if clip_min is None or clip_max is None:
                raise ValueError(
                    "Одно из значений clip_min и clip_max равно None, обрезка с одной стороны невозможны"
                )
            adv_X = torch.clamp(adv_X, clip_min, clip_max)
        
        return adv_X.detach()




