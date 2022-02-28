import pvlib
import pandas as pd


def extractTarget(effective_irradiance, cell_temperature, module, inverter):

    # dc, ac
    dc = pvlib.pvsystem.sapm(effective_irradiance, cell_temperature, module)
    ac = pvlib.inverter.sandia(dc['v_mp'], dc['p_mp'], inverter)
    annual_energy = ac.sum()

    # output: (ground truth) computed with pvlib
    # dc, ac
    target = pd.DataFrame()
    frames = [target, dc, ac]
    target = pd.concat(frames, axis=1)

    target.rename(columns={0: 'ac'}, inplace=True)

    return target, dc, ac, annual_energy
