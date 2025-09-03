/**
 * FlatBuffers Implementation for Legal AI
 * Optimized for WebAssembly and binary performance
 * Based on Google FlatBuffers JavaScript implementation
 */

// Constants for FlatBuffers
const SIZEOF_SHORT = 2;
const SIZEOF_INT = 4;
const FILE_IDENTIFIER_LENGTH = 4;

/**
 * FlatBuffers Builder for binary serialization
 */
class Builder {
  constructor(initial_size = 1024) {
    this.bb = new ByteBuffer(initial_size);
    this.space = initial_size;
    this.minalign = 1;
    this.vtable = null;
    this.vtable_in_use = 0;
    this.isNested = false;
    this.object_start = 0;
    this.vtables = [];
    this.vector_num_elems = 0;
    this.force_defaults = false;
  }

  /**
   * Start building an object with the given number of fields
   */
  startObject(numfields) {
    this.notNested();
    if (this.vtable == null) {
      this.vtable = [];
    }
    this.vtable_in_use = numfields;
    for (let i = 0; i < numfields; i++) {
      this.vtable[i] = 0;
    }
    this.isNested = true;
    this.object_start = this.offset();
  }

  /**
   * Finish building an object
   */
  endObject() {
    if (this.vtable == null || !this.isNested) {
      throw new Error('FlatBuffers: endObject called without startObject');
    }

    this.addInt32(0);
    const vtableloc = this.offset();

    // Trim trailing zeroes
    let i = this.vtable_in_use - 1;
    for (; i >= 0 && this.vtable[i] == 0; i--) {}
    const trimmed_size = i + 1;

    // Write vtable
    for (let j = trimmed_size - 1; j >= 0; j--) {
      this.addInt16(this.vtable[j] != 0 ? vtableloc - this.vtable[j] : 0);
    }

    const standard_fields = 2; // Object header
    this.addInt16((trimmed_size + standard_fields) * SIZEOF_SHORT);

    // Search for an existing vtable that matches
    let existing_vtable = 0;
    const vt1 = this.bb.bytes_.subarray(this.space, this.space + (trimmed_size + standard_fields) * SIZEOF_SHORT);

    outer: for (let j = 0; j < this.vtables.length; j++) {
      const vt2_offset = this.bb.capacity_ - this.vtables[j];
      const vt2 = this.bb.bytes_.subarray(vt2_offset, vt2_offset + vt1.length);
      
      if (vt1.length === vt2.length) {
        for (let k = 0; k < vt1.length; k++) {
          if (vt1[k] !== vt2[k]) {
            continue outer;
          }
        }
        existing_vtable = this.vtables[j];
        break;
      }
    }

    if (existing_vtable) {
      // Use existing vtable
      this.space = this.bb.capacity_ - vtableloc;
      this.bb.writeInt32(this.space, existing_vtable - vtableloc);
    } else {
      // Use new vtable
      this.vtables.push(this.offset());
      this.bb.writeInt32(this.bb.capacity_ - vtableloc, this.offset() - vtableloc);
    }

    this.isNested = false;
    return vtableloc;
  }

  /**
   * Finish building and return the buffer
   */
  finish(root_table, file_identifier) {
    const size_prefix = null;
    if (file_identifier) {
      this.prep(this.minalign, SIZEOF_INT + FILE_IDENTIFIER_LENGTH);
      if (file_identifier.length != FILE_IDENTIFIER_LENGTH) {
        throw new Error('FlatBuffers: file identifier must be length 4');
      }
      for (let i = FILE_IDENTIFIER_LENGTH - 1; i >= 0; i--) {
        this.addInt8(file_identifier.charCodeAt(i));
      }
    }
    this.prep(this.minalign, SIZEOF_INT);
    this.addOffset(root_table);
    if (size_prefix) {
      this.addInt32(this.bb.capacity_ - this.space);
    }
    this.bb.position_ = this.space;
  }

  /**
   * Get the finished buffer as Uint8Array
   */
  asUint8Array() {
    return this.bb.bytes_.subarray(this.bb.position_, this.bb.capacity_);
  }

  /**
   * Add a field to the current object
   */
  addFieldOffset(voffset, value, defaultValue) {
    if (this.force_defaults || value != defaultValue) {
      this.addOffset(value);
      this.slot(voffset);
    }
  }

  addFieldInt32(voffset, value, defaultValue) {
    if (this.force_defaults || value != defaultValue) {
      this.addInt32(value);
      this.slot(voffset);
    }
  }

  addFieldInt64(voffset, value, defaultValue) {
    if (this.force_defaults || value != defaultValue) {
      this.addInt64(value);
      this.slot(voffset);
    }
  }

  addFieldFloat32(voffset, value, defaultValue) {
    if (this.force_defaults || value != defaultValue) {
      this.addFloat32(value);
      this.slot(voffset);
    }
  }

  /**
   * Create a string
   */
  createString(s) {
    if (s === null || s === undefined) {
      return 0;
    }

    // Encode string to UTF-8
    const utf8 = [];
    for (let i = 0; i < s.length; i++) {
      let c = s.charCodeAt(i);
      if (c < 0x80) {
        utf8.push(c);
      } else if (c < 0x800) {
        utf8.push(0xC0 | (c >> 6));
        utf8.push(0x80 | (c & 0x3F));
      } else if (c < 0xD800 || c >= 0xE000) {
        utf8.push(0xE0 | (c >> 12));
        utf8.push(0x80 | ((c >> 6) & 0x3F));
        utf8.push(0x80 | (c & 0x3F));
      } else {
        // Surrogate pair
        i++;
        c = 0x10000 + (((c & 0x3FF) << 10) | (s.charCodeAt(i) & 0x3FF));
        utf8.push(0xF0 | (c >> 18));
        utf8.push(0x80 | ((c >> 12) & 0x3F));
        utf8.push(0x80 | ((c >> 6) & 0x3F));
        utf8.push(0x80 | (c & 0x3F));
      }
    }

    this.addInt8(0); // null terminator
    this.startVector(1, utf8.length, 1);
    this.bb.position_ -= utf8.length;
    for (let i = 0; i < utf8.length; i++) {
      this.bb.bytes_[this.bb.position_ + i] = utf8[i];
    }
    return this.endVector();
  }

  /**
   * Start building a vector
   */
  startVector(elem_size, num_elems, alignment) {
    this.notNested();
    this.vector_num_elems = num_elems;
    this.prep(SIZEOF_INT, elem_size * num_elems);
    this.prep(alignment, elem_size * num_elems);
  }

  /**
   * Finish building a vector
   */
  endVector() {
    this.addInt32(this.vector_num_elems);
    return this.offset();
  }

  // Internal methods
  notNested() {
    if (this.isNested) {
      throw new Error('FlatBuffers: Object serialization must not be nested');
    }
  }

  slot(voffset) {
    this.vtable[voffset] = this.offset();
  }

  offset() {
    return this.bb.capacity_ - this.space;
  }

  pad(byte_size) {
    for (let i = 0; i < byte_size; i++) {
      this.bb.writeInt8(--this.space, 0);
    }
  }

  prep(size, additional_bytes) {
    if (size > this.minalign) {
      this.minalign = size;
    }
    const align_size = (~(this.bb.capacity_ - this.space + additional_bytes) + 1) & (size - 1);
    while (this.space < align_size + additional_bytes + this.minalign) {
      const old_buf_size = this.bb.capacity_;
      this.bb = Builder.growByteBuffer(this.bb);
      this.space += this.bb.capacity_ - old_buf_size;
    }
    this.pad(align_size);
  }

  addInt8(value) {
    this.bb.writeInt8(--this.space, value);
  }

  addInt16(value) {
    this.prep(SIZEOF_SHORT, 0);
    this.bb.writeInt16(this.space -= SIZEOF_SHORT, value);
  }

  addInt32(value) {
    this.prep(SIZEOF_INT, 0);
    this.bb.writeInt32(this.space -= SIZEOF_INT, value);
  }

  addInt64(value) {
    this.prep(8, 0);
    this.bb.writeInt64(this.space -= 8, value);
  }

  addFloat32(value) {
    this.prep(SIZEOF_INT, 0);
    this.bb.writeFloat32(this.space -= SIZEOF_INT, value);
  }

  addOffset(offset) {
    this.prep(SIZEOF_INT, 0);
    this.addInt32(this.offset() - offset + SIZEOF_INT);
  }

  static growByteBuffer(bb) {
    const old_buf_size = bb.capacity_;
    if (old_buf_size & 0xC0000000) {
      throw new Error('FlatBuffers: cannot grow buffer beyond 2 gigabytes');
    }
    const new_buf_size = old_buf_size << 1;
    const nbb = new ByteBuffer(new_buf_size);
    nbb.position_ = nbb.capacity_ - old_buf_size;
    nbb.bytes_.set(bb.bytes_, nbb.position_);
    return nbb;
  }
}

/**
 * ByteBuffer for binary data manipulation
 */
class ByteBuffer {
  constructor(bytes_or_size) {
    if (typeof bytes_or_size === 'number') {
      this.bytes_ = new Uint8Array(bytes_or_size);
      this.position_ = 0;
    } else {
      this.bytes_ = bytes_or_size;
      this.position_ = 0;
    }
    this.capacity_ = this.bytes_.length;
  }

  capacity() {
    return this.capacity_;
  }

  position() {
    return this.position_;
  }

  setPosition(position) {
    this.position_ = position;
  }

  // Read methods
  readInt8(offset) {
    return this.readUint8(offset) << 24 >> 24;
  }

  readUint8(offset) {
    return this.bytes_[offset];
  }

  readInt16(offset) {
    return this.readUint16(offset) << 16 >> 16;
  }

  readUint16(offset) {
    return this.bytes_[offset] | (this.bytes_[offset + 1] << 8);
  }

  readInt32(offset) {
    return this.bytes_[offset] | 
           (this.bytes_[offset + 1] << 8) | 
           (this.bytes_[offset + 2] << 16) | 
           (this.bytes_[offset + 3] << 24);
  }

  readUint32(offset) {
    return this.readInt32(offset) >>> 0;
  }

  readInt64(offset) {
    return BigInt(this.readUint32(offset)) | 
           (BigInt(this.readUint32(offset + 4)) << 32n);
  }

  readFloat32(offset) {
    const view = new DataView(this.bytes_.buffer, this.bytes_.byteOffset + offset, 4);
    return view.getFloat32(0, true);
  }

  readFloat64(offset) {
    const view = new DataView(this.bytes_.buffer, this.bytes_.byteOffset + offset, 8);
    return view.getFloat64(0, true);
  }

  // Write methods
  writeInt8(offset, value) {
    this.bytes_[offset] = value;
  }

  writeUint8(offset, value) {
    this.bytes_[offset] = value;
  }

  writeInt16(offset, value) {
    this.bytes_[offset] = value;
    this.bytes_[offset + 1] = value >> 8;
  }

  writeUint16(offset, value) {
    this.bytes_[offset] = value;
    this.bytes_[offset + 1] = value >> 8;
  }

  writeInt32(offset, value) {
    this.bytes_[offset] = value;
    this.bytes_[offset + 1] = value >> 8;
    this.bytes_[offset + 2] = value >> 16;
    this.bytes_[offset + 3] = value >> 24;
  }

  writeUint32(offset, value) {
    this.writeInt32(offset, value);
  }

  writeInt64(offset, value) {
    this.writeInt32(offset, Number(value & 0xFFFFFFFFn));
    this.writeInt32(offset + 4, Number(value >> 32n));
  }

  writeFloat32(offset, value) {
    const view = new DataView(this.bytes_.buffer, this.bytes_.byteOffset + offset, 4);
    view.setFloat32(0, value, true);
  }

  writeFloat64(offset, value) {
    const view = new DataView(this.bytes_.buffer, this.bytes_.byteOffset + offset, 8);
    view.setFloat64(0, value, true);
  }

  // String operations
  __string(offset, encoding = 'utf8') {
    offset += this.readInt32(offset);
    const length = this.readInt32(offset);
    let result = '';
    offset += 4;
    
    if (encoding === 'utf8') {
      for (let i = 0; i < length; i++) {
        const byte = this.bytes_[offset + i];
        if (byte < 0x80) {
          result += String.fromCharCode(byte);
        } else {
          // Handle multi-byte UTF-8 sequences
          let charCode = 0;
          let bytesToRead = 0;
          
          if ((byte & 0xE0) === 0xC0) {
            charCode = byte & 0x1F;
            bytesToRead = 1;
          } else if ((byte & 0xF0) === 0xE0) {
            charCode = byte & 0x0F;
            bytesToRead = 2;
          } else if ((byte & 0xF8) === 0xF0) {
            charCode = byte & 0x07;
            bytesToRead = 3;
          }
          
          for (let j = 0; j < bytesToRead && i + j + 1 < length; j++) {
            charCode = (charCode << 6) | (this.bytes_[offset + i + j + 1] & 0x3F);
          }
          
          result += String.fromCharCode(charCode);
          i += bytesToRead;
        }
      }
    } else {
      // Binary mode
      for (let i = 0; i < length; i++) {
        result += String.fromCharCode(this.bytes_[offset + i]);
      }
    }
    
    return result;
  }

  // Indirect access
  __indirect(offset) {
    return offset + this.readInt32(offset);
  }

  __vector_len(offset) {
    return this.readInt32(this.__indirect(offset));
  }

  __vector(offset) {
    return offset + this.readInt32(offset) + 4;
  }
}

/**
 * Legal AI specific FlatBuffer schemas
 */

// Message types for legal AI communication
const MessageType = {
  PROMPT_REQUEST: 0,
  COMPLETION_RESPONSE: 1,
  EMBEDDING_REQUEST: 2,
  EMBEDDING_RESPONSE: 3,
  TRAINING_DATA: 4,
  ERROR_MESSAGE: 5
};

/**
 * Create a prompt request FlatBuffer
 */
function createPromptRequest(builder, prompt, context = [], options = {}) {
  // Create strings
  const promptOffset = builder.createString(prompt);
  const contextOffset = builder.createString(JSON.stringify(context));
  const optionsOffset = builder.createString(JSON.stringify(options));
  
  // Build the message
  builder.startObject(5);
  builder.addFieldInt32(0, MessageType.PROMPT_REQUEST, 0); // message_type
  builder.addFieldOffset(1, promptOffset, 0); // prompt
  builder.addFieldOffset(2, contextOffset, 0); // context
  builder.addFieldOffset(3, optionsOffset, 0); // options
  builder.addFieldInt64(4, BigInt(Date.now()), 0n); // timestamp
  
  return builder.endObject();
}

/**
 * Create an embedding request FlatBuffer
 */
function createEmbeddingRequest(builder, text, model = 'nomic-embed-text') {
  const textOffset = builder.createString(text);
  const modelOffset = builder.createString(model);
  
  builder.startObject(4);
  builder.addFieldInt32(0, MessageType.EMBEDDING_REQUEST, 0);
  builder.addFieldOffset(1, textOffset, 0);
  builder.addFieldOffset(2, modelOffset, 0);
  builder.addFieldInt64(3, BigInt(Date.now()), 0n);
  
  return builder.endObject();
}

/**
 * Create training data FlatBuffer
 */
function createTrainingData(builder, episodes) {
  const dataOffset = builder.createString(JSON.stringify(episodes));
  
  builder.startObject(3);
  builder.addFieldInt32(0, MessageType.TRAINING_DATA, 0);
  builder.addFieldOffset(1, dataOffset, 0);
  builder.addFieldInt64(2, BigInt(Date.now()), 0n);
  
  return builder.endObject();
}

// Export for use in service workers and Node.js
const flatbuffers = {
  Builder,
  ByteBuffer,
  MessageType,
  createPromptRequest,
  createEmbeddingRequest,
  createTrainingData
};

if (typeof module !== 'undefined' && module.exports) {
  module.exports = flatbuffers;
} else if (typeof self !== 'undefined') {
  self.flatbuffers = flatbuffers;
}

console.log('📦 FlatBuffers implementation loaded successfully');