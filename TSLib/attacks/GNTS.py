import torch

class GNTS():
    def __init__(self, device):
        """
        Методика состязательной атаки методом добавления на временной ряд Гаусовского шума. 
       
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
            
            # Проверка корректности введеных значений clip_min и clip_max
            if clip_min is not None and clip_max is not None:
                if clip_min > clip_max:
                    raise ValueError(
                        "clip_min должен быть меньше или равен clip_max, clip_min={}  clip_max={}".format(
                            clip_min, clip_max
                        )
                    )

            # Перенос тензора X на вычислительное устройство
            X = X.clone().detach().to(self.device)
            
            # Создание тензора содержащего гаусовский шум
            gn = torch.randn_like(X).sign()

            # Обнуление элементов шумового тензора не подлежащих атаке
            if num_elem is not None:
                gn[:, :-num_elem, :] = 0

            # Создание атакованного экземпляра временного ряда
            adv_X = X + eps * gn

            if (clip_min is not None) or (clip_max is not None):
                if clip_min is None or clip_max is None:
                    raise ValueError(
                        "Одно из значений clip_min и clip_max равно None, обрезка с одной стороны невозможны"
                    )
                adv_X = torch.clamp(adv_X, clip_min, clip_max)
            
            return adv_X.detach()




