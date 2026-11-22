import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { Markdown } from "../src/components/rent/Markdown";

const sample = `**Typical rent in Lekki** depends on size [Source 1].

| Property type | Note |
|---|---|
| All flats | **middle** asking price 【Source 2】 |
| 2-bedroom | varies |

\\*Monthly figures are yearly divided by 12.

What to expect:
- Most listings fall in a wide range
- Negotiate using the *median*
1. First
2. Second`;

describe("markdown", () => {
  it("renders tables, lists, bold, italics and citations without raw symbols", () => {
    const { container } = render(<Markdown text={sample} sources={[{ title: "One", url: "https://a.example" }, { title: "Two", url: "https://b.example" }]} />);
    expect(container.querySelector("table")).toBeTruthy();
    expect(container.querySelectorAll("tbody tr").length).toBe(2);
    expect(container.querySelector("ul")?.children.length).toBe(2);
    expect(container.querySelector("ol")?.children.length).toBe(2);
    expect(screen.getByText("Typical rent in Lekki").closest("strong")).toBeTruthy();
    expect(screen.getByText("median").tagName).toBe("EM");
    for (const raw of ["|---", "**", "【", "\\"]) expect(container.textContent).not.toContain(raw);
    const cite = screen.getAllByRole("link").map((a) => a.getAttribute("href"));
    expect(cite).toEqual(["https://a.example", "https://b.example"]);
  });

  it("never renders HTML from the text", () => {
    const { container } = render(<Markdown text={'<img src=x onerror="alert(1)"> hello'} />);
    expect(container.querySelector("img")).toBeNull();
    expect(container.textContent).toContain("<img");
  });
});

describe("markdown links", () => {
  it("only links to cited sources", () => {
    const { container } = render(<Markdown text={"See [this](https://evil.example/login) and [that](https://a.example)."} sources={[{ title: "A", url: "https://a.example" }]} />);
    const hrefs = [...container.querySelectorAll("a")].map((a) => a.getAttribute("href"));
    expect(hrefs).toEqual(["https://a.example"]);
    expect(container.textContent).toContain("this");
  });
});
