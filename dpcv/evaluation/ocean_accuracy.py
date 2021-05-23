


def test_regresor_per_factor(index=0, show=False):
    # Test como regresor
    tolerance_i = 0.1
    X = ds.data_test['X']
    Y = ds.data_test['Y']
    out = classifier.predict(X)

    acc = 0
    diff = 0
    for i in range(out.shape[0]):
        diff = diff + (1 - abs(Y[i][index] - out[i][index]))

    acc = diff / (out.shape[0] * 1)
    acc = acc * 100

    if show:
        print('Acc CL', acc)
    acc1 = acc

    acc = 0
    diff = 0
    for i in range(out.shape[0]):
        diff = abs(Y[i][index] - out[i][index])

        if diff <= tolerance_i:
            acc = acc + 1

    acc = acc / (out.shape[0] * 1)
    acc = acc * 100

    if show:
        print('Acc uno tol', acc)

    return (acc1, acc)


def get_global_evaluation(show=False):
    prom_cl = 0
    prom_to = 0

    if show:
        print('O')
    (a1, a2) = test_regresor_per_factor(index=0, show=show)
    prom_cl += a1
    prom_to += a2

    if show:
        print('\nC')
    (a1, a2) = test_regresor_per_factor(index=1, show=show)
    prom_cl += a1
    prom_to += a2

    if show:
        print('\nE')
    (a1, a2) = test_regresor_per_factor(index=2, show=show)
    prom_cl += a1
    prom_to += a2

    if show:
        print('\nA')
    (a1, a2) = test_regresor_per_factor(index=3, show=show)
    prom_cl += a1
    prom_to += a2

    if show:
        print('\nN')
    (a1, a2) = test_regresor_per_factor(index=4, show=show)
    prom_cl += a1
    prom_to += a2

    prom_cl = prom_cl / 5
    prom_to = prom_to / 5

    if show:
        print('\nPromedio')
        print('ChaLearn', prom_cl)
        print('Uno tol', prom_to)

    return prom_cl


get_global_evaluation(show=True)