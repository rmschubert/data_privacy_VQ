import numpy as np


class PlainGLVQ:
    def __call__(
        self,
        prototypes,
        pc,
        sample,
        sc=None,
        lr=1.0,
        mode="train",
        with_indices=False,
    ):
        if mode == "train":
            assert sc is not None, f"sample class cannot be None when mode is {mode}"

            parameters = prototypes.copy()
            indices = range(parameters.shape[0])

            distances = self.belongingness(parameters, sample)
            decision = self.decision_function(distances, parameters, pc, sc, indices)

            mags, gradients = self.gradients(lr, decision, sample)
            updated_prototypes = self.update(parameters, decision, gradients)

            if with_indices:
                return mags, gradients, updated_prototypes, decision[-1]
            else:
                return mags, gradients, updated_prototypes
        else:
            parameters = prototypes.copy()
            distances = self.belongingness(parameters, sample)
            return self.assignment(distances, pc)

    def belongingness(self, prototypes, sample):
        x = np.tile(sample, (prototypes.shape[0], 1))
        diff = x - prototypes
        return np.sum(diff**2, -1)

    def class_filter(self, pc, sc):
        class_match = [c == sc for c in pc]
        class_mismatch = [c != sc for c in pc]

        return class_match, class_mismatch

    def decision_function(self, distances, prototypes, pc, sc, indices):
        mp, mm = self.class_filter(pc, sc)

        wpi = np.argmin(distances[mp])
        wp = prototypes[mp][wpi]
        wmi = np.argmin(distances[mm])
        wm = prototypes[mm][wmi]

        dp = distances[mp][wpi]
        dm = distances[mm][wmi]

        wp_inds = [i[0] for i in zip(indices, mp) if i[1]]
        wm_inds = [i[0] for i in zip(indices, mm) if i[1]]
        sel_prototypes = [wp_inds[wpi], wm_inds[wmi]]

        return (wp, wm, dp, dm, sel_prototypes)

    def sig(self, wpd, wmd, gamma=0.25):
        mu = (wpd - wmd) / (wpd + wmd)
        return 1 / (1 + np.exp(-mu * gamma))

    def gradients(self, lr, decision, sample):
        wp, wm, wpd, wmd, _ = decision
        sig = self.sig(wpd, wmd)

        magp = lr * sig * (1 - sig) * 4 * (wmd / (wpd + wmd) ** 2)
        magm = lr * sig * (1 - sig) * 4 * (wpd / (wpd + wmd) ** 2)

        wpg = -magp * (sample - wp)
        wmg = magm * (sample - wm)

        return (magp, -magm), (wpg, wmg)

    def update(self, prototypes, decision, gradients):
        new_prototypes = prototypes.copy()
        wp, wm, _, _, inds = decision

        wpg, wmg = gradients
        wpt1 = wp - wpg
        wmt1 = wm - wmg

        new_prototypes[inds[0]] = wpt1
        new_prototypes[inds[1]] = wmt1

        return new_prototypes

    def assignment(self, distances, pc):
        winner = np.argmin(distances)
        assignment = pc[winner]
        return assignment
