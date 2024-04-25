import matplotlib.pyplot as plt
import numpy as np


def adam_opt(grads, moms, t, a=0.001, b1=0.9, b2=0.999, eps=1e-8):
    ## ADAM optimizer - hyperparameters taken from: paper-source

    new_grads = []
    new_moms = []
    for g, ms in zip(grads, moms):
        mt = b1 * ms[0] + (1 - b1) * g
        vt = b2 * ms[1] + (1 - b2) * g**2
        c_mt = mt / (1 - b1**t)
        c_vt = vt / (1 - b2**t)

        g_mom = a * (c_mt / (np.sqrt(c_vt) + eps))

        new_grads.append(g_mom)

        new_moms.append(tuple([mt, vt]))

    return new_grads, new_moms


def plot_protos(p_t):
    ## plot updated prototypes

    n = p_t.shape[0]

    f, axarr = plt.subplots(1, n)
    for i in range(n):
        axarr[i].imshow(p_t[i, :].reshape((28, 28)))
        axarr[i].axis("off")
        axarr[i].set_title(f"p{i + 1}")

    plt.tight_layout()
    plt.show()
    plt.close()


def find_updated_prototypes(p_t, p_t1):
    ## find updated prototypes by considering
    ## p(t + 1) - p(t)

    diff = p_t1 - p_t

    mask = np.any(np.abs(diff) > 0, axis=1)
    updated_protos_t = p_t[mask]
    updated_protos_t1 = p_t1[mask]

    imp_diffs = updated_protos_t1 - updated_protos_t

    # plot_protos(updated_protos_t)

    return updated_protos_t, updated_protos_t1, imp_diffs


def find_matching_gradient(gradients, deltas):
    ## find the gradients to the corresponding
    ## found prototypes

    matching_gradients = []
    inds = np.arange(len(gradients))
    signs = []
    for d in deltas:
        distances_p = []
        distances_m = []
        gs = np.asarray(gradients).copy()
        for g in gs:
            ## we do not know the sign for the update
            ## so we need to consider both directions

            distp = np.sum((d + g) ** 2)
            distm = np.sum((d - g) ** 2)

            distances_p.append(distp)
            distances_m.append(distm)

        ## find the matching gradient and sign for the current delta
        distances = list(zip(distances_p, distances_m))
        distances_p.extend(distances_m)
        all_distances = np.array(distances_p)
        match_ind = [
            i
            for i, pair in enumerate(distances)
            if pair[0] == all_distances.min() or pair[1] == all_distances.min()
        ]

        ## sign
        ## -1 corresponding to d = -g (gradient descent)
        ## +1 corresponding to d = +g (gradient ascent)
        sign = -1 if distances[match_ind[-1]][0] == all_distances.min() else 1
        signs.append(sign)

        ## gradient
        g_ind = int(inds[match_ind])
        matching_gradients.append(g_ind)

        ## avoid multiple assignments - filter out already assigned gradients
        grad_mask = [bool(i != g_ind) for i in inds]
        gradients = np.asarray(gradients)[grad_mask, :].copy()
        inds = inds[grad_mask].copy()

    return matching_gradients, signs


def attack_vq(gradients, prototypes_t, prototypes_t1, steps=100000, tol=1e-8):
    ## runs the actual attack
    ## early stop could be implemented by using obj < tol
    ## we here run for the fixed number of steps
    p_t, _, deltas = find_updated_prototypes(prototypes_t, prototypes_t1)

    if not np.any(deltas):
        return None

    grad_inds, signs = find_matching_gradient(gradients, deltas)
    results = []
    for i, _ in enumerate(deltas):
        gi = grad_inds[i]
        if p_t.shape[0] > 1:
            p = p_t[i, :]
            s = signs[i]
        else:
            p = p_t

        ## use assigned gradients and
        ## initialize data vector x and
        ## gain b randomly
        g = gradients[gi]
        x = np.random.rand(*p.shape)
        b = np.random.rand(1)

        moms = [np.random.rand(2, *parameter.shape) for parameter in [b, x]]

        for j in range(1, steps):
            ## update vector v based on the generic LR
            v = b * (x - p)

            ## squared euclidean distance as objective
            ## to be minimized between v and assigned gradient g
            obj = np.inner(g - v, g - v)

            ## gradients of obj wrt to gain b and vector x
            grad_b = -2 * np.inner(g - v, x - p)
            grad_x = -2 * b * (g - v)

            ## run adam
            grads, moms = adam_opt([grad_b, grad_x], moms, j)

            ## update the parameters
            b = b - grads[0]
            x = x - grads[1]

        ## results
        results.append([obj, s, b, x])

        ## It is likey that the best result for GLVQ
        ## relates to b < 0 for s = -1, since the sign
        ## of the gradient g^+ for w^+ is negative and corresponds to
        ## s * g^+ >= 0 for gradient descent learning, i.e.
        ## g^+ = b * (x - p^+) for b < 0 and s = -1 in the approximation.
        ## In detail, an attacker could guess that attraction-repelling is used
        ## if the estimated b's differ in sign and
        ## vice versa if b > 0 (or b < 0) for **all** found updated prototypes it is
        ## likely that attraction-repelling is not being used/apparent.
        ## Consequently, an attacker could take an result of b < 0 and s = -1
        ## and concentrate only on this approximation since this relates to attraction
        ## and in GLVQ only the prototypes with a matching class are attracted,
        ## thus also exposing the class resemblance.
    return results
