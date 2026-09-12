import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { SmilesInput } from "./SmilesInput";

describe("SmilesInput", () => {
  it("renders the text input", () => {
    render(<SmilesInput value="" onChange={() => {}} onSubmit={() => {}} disabled={false} />);
    expect(screen.getByLabelText("SMILES string")).toBeInTheDocument();
  });

  it("calls onChange when typing", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    render(<SmilesInput value="" onChange={onChange} onSubmit={() => {}} disabled={false} />);
    await user.type(screen.getByLabelText("SMILES string"), "CCO");
    expect(onChange).toHaveBeenCalledTimes(3);
    expect(onChange).toHaveBeenLastCalledWith("O");
  });

  it("calls onSubmit when the Predict button is clicked (valid input)", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    render(<SmilesInput value="CC(=O)Oc1ccccc1C(=O)O" onChange={() => {}} onSubmit={onSubmit} disabled={false} />);
    await user.click(screen.getByRole("button", { name: "Predict" }));
    expect(onSubmit).toHaveBeenCalledTimes(1);
  });

  it("calls onSubmit when pressing Enter in the input", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    render(<SmilesInput value="CCO" onChange={() => {}} onSubmit={onSubmit} disabled={false} />);
    screen.getByLabelText("SMILES string").focus();
    await user.keyboard("{Enter}");
    expect(onSubmit).toHaveBeenCalledTimes(1);
  });

  it("disables the input and button when disabled is true", () => {
    render(<SmilesInput value="CCO" onChange={() => {}} onSubmit={() => {}} disabled={true} />);
    expect(screen.getByLabelText("SMILES string")).toBeDisabled();
    expect(screen.getByRole("button", { name: "Predict" })).toBeDisabled();
  });

  it("disables the submit button when the value is blank (invalid/empty input)", () => {
    render(<SmilesInput value="   " onChange={() => {}} onSubmit={() => {}} disabled={false} />);
    expect(screen.getByRole("button", { name: "Predict" })).toBeDisabled();
  });
});
