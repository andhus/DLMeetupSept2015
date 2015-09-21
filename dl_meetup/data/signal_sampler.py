from __future__ import division, print_function

from abc import abstractmethod
import numpy as np


class SignalSampler(object):

    def __init__(
        self,
        n_time_steps
    ):
        self.n_time_steps = n_time_steps

    def samples(
        self,
        n_samples
    ):
        """
        Parmaters
        ---------
        :param n_samples: int

        Returns
        -------
        :return: samples_: numpy.ndarray(shape=(n_samples, self.n_time_steps))
        """
        sample_arrays = map(
            lambda _: self._sample(),
            range(n_samples)
        )
        samples_ = np.vstack(sample_arrays)

        return samples_

    @abstractmethod
    def _sample(self):
        """
        Returns
        :return: sample_ : numpy.ndarray(shape=(self.n_time_steps))
        """
        pass


class Sine(SignalSampler):
    """ Generates random sine signals. Parameters are sampled uniformly
    from specified ranges.

    Parameters
    ----------
    :param amplitude_range: (float, float)
    :param period_range: (float, float)
    :param phase_range: (float, float)

    """
    def __init__(
        self,
        n_time_steps,
        amplitude_range,
        period_range,
        phase_range,
    ):
        super(Sine, self).__init__(n_time_steps=n_time_steps)
        self.amplitude_range = amplitude_range
        self.period_range = period_range
        self.phase_range = phase_range

    def _sample(self):
        amplitude = np.random.uniform(*self.amplitude_range)
        period = np.random.uniform(*self.period_range)
        phase = np.random.uniform(*self.phase_range)
        sample_ = self._sine(
            amplitude,
            period,
            phase
        )

        return sample_

    def _sine(
        self,
        amplitude,
        period,
        phase
    ):
        time = np.arange(self.n_time_steps)
        sine = amplitude * np.sin(time*2*np.pi/period - phase)

        return sine


class Line(SignalSampler):
    """ Generates random "lines". Start and end value are sampled uniformly
    from specified ranges.

    Parameters
    ----------
    :param start_range: (float, float)
    :param end_range: (float, float)

    """
    def __init__(
        self,
        n_time_steps,
        start_range,
        end_range,
    ):
        super(Line, self).__init__(n_time_steps=n_time_steps)
        self.start_range = start_range
        self.end_range = end_range

    def _sample(self):
        start = np.random.uniform(*self.start_range)
        end = np.random.uniform(*self.end_range)
        sample_ = self._line(
            start,
            end
        )

        return sample_

    def _line(
        self,
        start,
        end,
    ):

        line = np.linspace(
            start=start,
            stop=end,
            num=self.n_time_steps
        )

        return line


class Square(SignalSampler):
    """ Generates random "square" signals.

    Parameters
    ----------
    :param start_range: (float, float)
    :param end_range: (float, float)

    """
    def __init__(
        self,
        n_time_steps,
        amplitude_range,
        off_duration_range,
        on_duration_range,
        duration_variation_range=(0, 0),
    ):
        super(Square, self).__init__(n_time_steps=n_time_steps)
        self.amplitude_range = amplitude_range
        self.off_duration_range = off_duration_range
        self.on_duration_range = on_duration_range
        self.duration_variation_range = duration_variation_range

    def _sample(self):
        amplitude = np.random.uniform(*self.amplitude_range)
        off_duration = np.random.uniform(*self.off_duration_range)
        on_duration = np.random.uniform(*self.on_duration_range)
        sample_ = self._square_signal(
            amplitude,
            off_duration,
            on_duration
        )

        return sample_

    def _square_signal(
        self,
        amplitude,
        off_duration,
        on_duration
    ):
        t = 0
        periods = []
        while t < self.n_time_steps + off_duration + on_duration:
            off_dur_one = int(
                off_duration + self._sample_period_variation()
            )
            periods.append(
                np.zeros(off_dur_one)
            )
            on_dur_one = int(
                on_duration + self._sample_period_variation()
            )
            periods.append(
                amplitude * np.ones(on_dur_one)
            )
            t += off_dur_one + on_dur_one

        square_signal_long = np.concatenate(
            periods
        )
        start = int(
            np.random.uniform(
                len(square_signal_long) - self.n_time_steps
            )
        )
        square_signal = square_signal_long[start:start+self.n_time_steps]

        return square_signal

    def _sample_period_variation(
        self,
    ):
        return np.random.uniform(
            self.duration_variation_range[0],
            self.duration_variation_range[1]
        )
