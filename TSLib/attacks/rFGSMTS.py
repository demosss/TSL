import torch

class RFGSMTS():
    def __init__(self, model, loss_fn, device):
        """
        Методика состязательной атаки методом random FSGM на временной ряд. 

        параметр: model - модель машинного обучения

        параметр: loss_fn - функция ошибки
        
        парамерт: device - вычислительное устройство (cpu/cuda)
        """
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.device = device




    def attack(self, X, y, eps=0.05, num_steps=10, alpha=None, clip_min=None, clip_max=None, targeted = False, num_elem = None):
        """
        Функция реализующая атаку

        параметр: X - образец (образцы) временного ряда подлежащий(ие) атаке
        
        параметр: y - ожидаемый (истинный) результат работы модели

        параметр: eps - величина внесенных возмущений в исходный временной ряд

        параметр: num_steps - количество шагов

        параметр: alpha - величина внесенных возмущений в исходный временной ряд за один шаг

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
        
        if eps == 0:
            return X
        
        # Проверка корректности выбранной величины steps
        if num_steps < 0:
            raise ValueError(
                "num_steps должен быть больше или равен 0, num_steps {} не допустим".format(num_steps)
            )
        
        # Определение величины alpha в случае None
        if alpha is None:
            alpha = eps / num_steps 
        
        # Проверка корректности выбранной величины alpha
        if alpha < 0:
            raise ValueError(
                "alpha должен быть больше или равен 0, alpha {} не допустим".format(alpha)
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
        X = X.to(self.device)
        y = y.clone().detach().to(self.device)

        X_adv = X + alpha * torch.randn_like(X).sign()
        X_adv = torch.clamp(X_adv, X - eps, X + eps).detach()



        

        for _ in range (num_steps):
            grad = self.return_grad(X_adv, y, self.model, self.loss_fn, targeted)

            # Обнуление градиентов элементов временного ряда не подлежащих атаке
            if num_elem is not None:
                grad[:, :-num_elem, :] = 0

            # Создание атакованного экземпляра временного ряда
            X_adv = X_adv.detach() + alpha * grad.sign()
            delta = torch.clamp(X_adv - X, -eps, eps)
            X_adv = torch.clamp(X_adv + delta, X - eps, X + eps).detach()

        if (clip_min is not None) or (clip_max is not None):
            if clip_min is None or clip_max is None:
                raise ValueError(
                    "Одно из значений clip_min и clip_max равно None, обрезка с одной стороны невозможны"
                )
            X_adv = torch.clamp(X_adv, clip_min, clip_max)
        
        return X_adv

    
    def return_grad(self, X_a, y, model, loss_fn, targeted):
        model.train()
        X_a.requires_grad = True
        model.zero_grad()
        output = model(X_a)
        loss = loss_fn(output, y)
        
        if targeted:
            loss = -loss

        grad = torch.autograd.grad(loss, X_a, retain_graph=False, create_graph=False)[0]

        X_a.requires_grad = False
        model.eval()
        return grad