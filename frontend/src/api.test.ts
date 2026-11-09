import { describe, expect, it } from "vitest";
import { parseSSE } from "./api";

describe("parseSSE", () => {
  it("returns complete events and keeps a partial one as the remainder", () => {
    const buffer =
      'event: route\ndata: {"type":"route","route":"logs","plan":"p"}\n\n' +
      'event: token\ndata: {"type":"token","text":"Hel"}\n\n' +
      'event: token\ndata: {"type":"tok';
    const { events, rest } = parseSSE(buffer);
    expect(events).toEqual([
      { type: "route", route: "logs", plan: "p" },
      { type: "token", text: "Hel" },
    ]);
    expect(rest).toBe('event: token\ndata: {"type":"tok');
  });

  it("handles an empty buffer", () => {
    expect(parseSSE("")).toEqual({ events: [], rest: "" });
  });
});
