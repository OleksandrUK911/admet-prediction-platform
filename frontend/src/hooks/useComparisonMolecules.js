import { useCallback, useMemo } from "react";
import { useSearchParams } from "react-router-dom";
import { COMPARISON_MAX_MOLECULES } from "../admetConstants";

// State for the /compare page's list of SMILES - TODO_state_data_layer.md
// ("Керування станом порівняння кількох молекул"). Persisted in the URL
// query string (repeated `m` params, e.g. `?m=CCO&m=CC(=O)O`) rather than
// component state, so a comparison is shareable/bookmarkable - the same
// idea already used for History -> Predict's `location.state` prefill,
// taken one step further since a URL survives a page reload/share where
// router state doesn't.
export function useComparisonMolecules() {
  const [searchParams, setSearchParams] = useSearchParams();

  const molecules = useMemo(() => searchParams.getAll("m"), [searchParams]);

  const writeMolecules = useCallback(
    (list) => {
      const next = new URLSearchParams();
      list.forEach((smiles) => {
        if (smiles.trim()) next.append("m", smiles.trim());
      });
      setSearchParams(next, { replace: true });
    },
    [setSearchParams],
  );

  const setMoleculeAt = useCallback(
    (index, smiles) => {
      const next = [...molecules];
      next[index] = smiles;
      writeMolecules(next);
    },
    [molecules, writeMolecules],
  );

  const removeMoleculeAt = useCallback(
    (index) => {
      writeMolecules(molecules.filter((_, i) => i !== index));
    },
    [molecules, writeMolecules],
  );

  const canAddMore = molecules.length < COMPARISON_MAX_MOLECULES;

  return { molecules, setMoleculeAt, removeMoleculeAt, canAddMore };
}
