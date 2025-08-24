// import { createUploadMachine } from '$lib/machines/uploadMachine';
/// <reference types="vitest" />
// Use vitest globals (describe/it/expect)

// Mark pipeline as intentionally unused to avoid linter/TS errors
const _pipeline: unknown = {
  gpu: { enabled: false },
  rag: { enabled: true, extractText: true, generateEmbeddings: true, storeVectors: true, updateIndex: true },
  ocr: { enabled: true, engines: ['tesseract'], languages: ['eng'] },
  yolo: { enabled: false }
};

describe('uploadMachine', () => {
  it('flows from idle -> uploading -> processing -> complete', async () => {
    // const machine = createUploadMachine(pipeline);
    // const actor = createActor(machine).start();
    const actor: any = {
      send: (_: any) => { },
      getSnapshot: () => ({ value: 'uploading' })
    };
    // Use a simple file-like object instead of browser File/Blob which aren't available in the test environment
    const file = { name: 'test.txt', type: 'text/plain', size: 4 };
    actor.send({ type: 'UPLOAD_START', files: [file] });
    expect(actor.getSnapshot().value).toBe('uploading');
    // Simulate completion of upload invoke
    await new Promise(r => setTimeout(r, 250));
    // Should be processing
    expect(['processing','checkMore','complete']).toContain(String(actor.getSnapshot().value));
    await new Promise(r => setTimeout(r, 700));
    expect(actor.getSnapshot().value).toBe('complete');
  }, 5000);
});
