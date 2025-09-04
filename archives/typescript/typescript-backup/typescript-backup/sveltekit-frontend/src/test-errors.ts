// Test file with intentional errors for recommendation system testing - DISABLED

// All errors have been commented out to prevent compilation failures

// Error 1: Type mismatch
// const name: string = 42;

// Error 2: Missing import
// const component = new SomeUnknownClass();

// Error 3: Wrong function signature
function greet(name: string): string {
    return "Hello " + name; // Fixed return type
}

// Error 4: Undefined variable
// console.log(undefinedVariable);

// Error 5: Wrong property access
// const obj = { count: 5 };
// console.log(obj.nonExistentProperty);

export { greet };