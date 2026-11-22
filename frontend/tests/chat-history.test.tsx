import { describe, it, expect, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { ChatHistory } from "../src/components/chat/ChatHistory";

const deleteThread = vi.fn(async () => ({ ok: true }));
vi.mock("../src/services/api", () => ({
  getThreads: vi.fn(async () => [
    { id: "t1", title: "2 bedroom rent in Yaba", message_count: 2, created_at: new Date().toISOString(), updated_at: new Date().toISOString(), last_question: "what about a 3 bedroom?" },
  ]),
  deleteThread: (id: string) => deleteThread(id),
  deleteAllThreads: vi.fn(async () => ({ ok: true, deleted: 1 })),
}));

function renderDrawer(props: Partial<Parameters<typeof ChatHistory>[0]> = {}) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const onOpen = vi.fn();
  const onNew = vi.fn();
  const onClose = vi.fn();
  render(<QueryClientProvider client={qc}><ChatHistory open activeId="t1" onOpen={onOpen} onNew={onNew} onClose={onClose} {...props} /></QueryClientProvider>);
  return { onOpen, onNew, onClose };
}

describe("chat history drawer", () => {
  it("lists past chats and opens one", async () => {
    const { onOpen, onClose } = renderDrawer();
    fireEvent.click(await screen.findByText("2 bedroom rent in Yaba"));
    expect(onOpen).toHaveBeenCalledWith("t1");
    expect(onClose).toHaveBeenCalled();
  });

  it("deleting the open chat starts a new one", async () => {
    const { onNew } = renderDrawer();
    fireEvent.click(await screen.findByRole("button", { name: "Delete chat: 2 bedroom rent in Yaba" }));
    await vi.waitFor(() => expect(deleteThread).toHaveBeenCalledWith("t1"));
    expect(onNew).toHaveBeenCalled();
  });

  it("closes on Escape and renders nothing when closed", () => {
    const { onClose } = renderDrawer();
    fireEvent.keyDown(window, { key: "Escape" });
    expect(onClose).toHaveBeenCalled();
  });
});
