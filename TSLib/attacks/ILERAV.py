import torch

class ILERAV():
    def __init__(self, device):
        """
        Методика состязательной атаки методом Inversion of the last elements relative to the average value на временной ряд.
        
        парамерт: device - вычислительное устройство (cpu/cuda)
        """
        self.device = device




    def attack(self, X, eps=0.05, clip_min=None, clip_max=None, num_elem = None):
        """
        Функция реализующая атаку

        параметр: X - образец (образцы) временного ряда подлежащий(ие) атаке

        параметр: eps - величина внесенных возмущений в исходный временной ряд

        параметр: clip_min - минимальное значение атакованного экземпляра временного ряда 

        параметр: clip_max - максимальное значение атакованного экземпляра временного ряда

        параметр: num_elem - количество значимых коэффициентов автокорреляции временного ряда 
        (определяет количество последних элементов временного ряда, подлежащих атаке, None - атакуются все элементы)
        """

        # Проверка корректности выбранной величины eps
        if eps < 0:
            raise ValueError(
                "eps должен быть больше или равен 0, eps {} не допустим".format(eps)
            )
        
        # Определение величины num_elem в случае None
        if num_elem is None:
            num_elem = 1

        # Проверка корректности выбранной величины alpha
        if num_elem < 1:
            raise ValueError(
                "num_elem должен быть больше или равен 1, num_elem {} не допустим".format(num_elem)
            )
        
        X = X.clone().detach().to(self.device)
        X_adv = X.clone().detach().to(self.device)
        num_elem += 1
        
        for i in range(X.shape[0]):
            for j in range(X.shape[2]):
                let = X[i, -num_elem:, j]
                average = torch.mean(X[i, -num_elem:, j], 0)
                revers = 2 * average - let
                X_adv[i, -num_elem:, j] = revers
                
        X_adv = X_adv + eps * (X_adv - X).sign()

        if (clip_min is not None) or (clip_max is not None):
            if clip_min is None or clip_max is None:
                raise ValueError(
                    "Одно из значений clip_min и clip_max равно None, обрезка с одной стороны невозможны"
                )
            adv_X = torch.clamp(adv_X, clip_min, clip_max)
        
        return X_adv.detach()

        
        




        




