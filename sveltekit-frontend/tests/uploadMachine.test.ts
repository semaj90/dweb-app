import { createUploadMachine } from '$lib/machines/uploadMachine';
import { describe, expect, it } from 'vitest';
import { createActor } from 'xstate';

const pipeline: any = {
  gpu: { enabled: false },
  rag: { enabled: true, extractText: true, generateEmbeddings: true, storeVectors: true, updateIndex: true },
  ocr: { enabled: true, engines: ['tesseract'], languages: ['eng'] },
  yolo: { enabled: false }
};

describe('uploadMachine', () => {
  it('flows from idle -> uploading -> processing -> complete', async () => {
    const machine = createUploadMachine(pipeline);
    const actor = createActor(machine).start();
    const file = new File([new Blob(['test'])], 'test.txt');
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
