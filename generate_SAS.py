import ast
import re
import json
import os
import sys
import argparse
import shutil
from typing import Optional, List, Set, Dict, Any
import glob
from collections import defaultdict

class StructuredSkeletonGenerator(ast.NodeVisitor):
    """Generate code skeletons that preserve program structure, all original names, and return statements."""
    
    def __init__(self):
        self.reset_state()
        
    def reset_state(self):
        self.lines = []
        self.current_indent = 0
        self.imports = set()
        
    def indent(self):
        self.current_indent += 1

    def dedent(self):
        self.current_indent = max(self.current_indent - 1, 0)

    def add_line(self, line=""):
        indent_str = ' ' * 4 * self.current_indent
        self.lines.append(indent_str + line)

    def visit_Import(self, node):
        for alias in node.names:
            if alias.asname:
                self.imports.add(f"import {alias.name} as {alias.asname}")
            else:
                self.imports.add(f"import {alias.name}")

    def visit_ImportFrom(self, node):
        names = []
        for alias in node.names:
            if alias.asname:
                names.append(f"{alias.name} as {alias.asname}")
            else:
                names.append(alias.name)
        module = node.module or ""
        if node.level > 0:
            module = "." * node.level + module
        if module:
            self.imports.add(f"from {module} import {', '.join(names)}")

    def _format_arguments(self, args: ast.arguments) -> str:
        """Format function arguments with original names."""
        parts = []
        
        # Positional-only arguments
        pos_only = getattr(args, 'posonlyargs', [])
        for arg in pos_only:
            parts.append(arg.arg)
            
        # Normal arguments
        for arg in args.args:
            parts.append(arg.arg)
            
        # *args
        if args.vararg:
            parts.append(f"*{args.vararg.arg}")
            
        # Keyword-only arguments  
        for arg in args.kwonlyargs or []:
            parts.append(arg.arg)
            
        # **kwargs
        if args.kwarg:
            parts.append(f"**{args.kwarg.arg}")
            
        return ", ".join(parts)

    def visit_FunctionDef(self, node):
        """Visit function definition, preserving all names and return statements."""
        decorators = []
        for decorator in node.decorator_list:
            try:
                decorators.append(f"@{ast.unparse(decorator)}")
            except:
                decorators.append("@decorator")
                
        args = self._format_arguments(node.args)
        
        # Add decorators
        for decorator in decorators:
            self.add_line(decorator)
            
        # Function definition - KEEP ALL NAMES
        self.add_line(f"def {node.name}({args}):")
        self.indent()
        
        # Visit function body to preserve structure
        for item in node.body:
            self.visit(item)
            
        self.dedent()
        self.add_line()

    def visit_AsyncFunctionDef(self, node):
        """Visit async function definition."""
        # Convert to regular function def for processing
        regular_node = ast.FunctionDef(
            name=node.name,
            args=node.args,
            body=node.body,
            decorator_list=node.decorator_list,
            returns=node.returns,
            type_comment=node.type_comment
        )
        self.visit_FunctionDef(regular_node)

    def visit_ClassDef(self, node):
        """Visit class definition - KEEP ALL NAMES."""
        bases = []
        for base in node.bases:
            try:
                bases.append(ast.unparse(base))
            except:
                bases.append("...")
                
        bases_str = f"({', '.join(bases)})" if bases else ""
        
        # Class decorators
        for decorator in node.decorator_list:
            try:
                self.add_line(f"@{ast.unparse(decorator)}")
            except:
                self.add_line("@decorator")
                
        # Class definition - KEEP ALL NAMES
        self.add_line(f"class {node.name}{bases_str}:")
        self.indent()
        
        # Visit class body
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.visit(item)
            elif isinstance(item, ast.Assign):
                # Handle class variables - KEEP ALL NAMES
                targets = ", ".join(ast.unparse(t) for t in item.targets)
                self.add_line(f"{targets} = ...")
            else:
                self.add_line("...")
                
        self.dedent()
        self.add_line()

    def visit_For(self, node):
        """Preserve for loop structure with original variable names."""
        target = ast.unparse(node.target)
        iter_expr = ast.unparse(node.iter) if hasattr(node, 'iter') else "..."
        self.add_line(f"for {target} in {iter_expr}:")
        self.indent()
        for item in node.body:
            self.visit(item)
        self.dedent()
        
        if node.orelse:
            self.add_line("else:")
            self.indent()
            for item in node.orelse:
                self.visit(item)
            self.dedent()

    def visit_While(self, node):
        """Preserve while loop structure."""
        test_expr = ast.unparse(node.test) if hasattr(node, 'test') else "..."
        self.add_line(f"while {test_expr}:")
        self.indent()
        for item in node.body:
            self.visit(item)
        self.dedent()
        
        if node.orelse:
            self.add_line("else:")
            self.indent()
            for item in node.orelse:
                self.visit(item)
            self.dedent()

    def visit_If(self, node):
        """Preserve if-elif-else structure."""
        test_expr = ast.unparse(node.test) if hasattr(node, 'test') else "..."
        self.add_line(f"if {test_expr}:")
        self.indent()
        for item in node.body:
            self.visit(item)
        self.dedent()
        
        # Handle elif/else chains
        current = node
        while current.orelse and len(current.orelse) == 1 and isinstance(current.orelse[0], ast.If):
            current = current.orelse[0]
            test_expr = ast.unparse(current.test) if hasattr(current, 'test') else "..."
            self.add_line(f"elif {test_expr}:")
            self.indent()
            for item in current.body:
                self.visit(item)
            self.dedent()
            
        if current.orelse and not (len(current.orelse) == 1 and isinstance(current.orelse[0], ast.If)):
            self.add_line("else:")
            self.indent()
            for item in current.orelse:
                self.visit(item)
            self.dedent()

    def visit_Try(self, node):
        """Preserve try-except structure."""
        self.add_line("try:")
        self.indent()
        for item in node.body:
            self.visit(item)
        self.dedent()
        
        for handler in node.handlers:
            if handler.type:
                type_str = ast.unparse(handler.type)
            else:
                type_str = ""
            name = f" as {handler.name}" if handler.name else ""
            self.add_line(f"except {type_str}{name}:")
            self.indent()
            for item in handler.body:
                self.visit(item)
            self.dedent()
            
        if node.orelse:
            self.add_line("else:")
            self.indent()
            for item in node.orelse:
                self.visit(item)
            self.dedent()
            
        if node.finalbody:
            self.add_line("finally:")
            self.indent()
            for item in node.finalbody:
                self.visit(item)
            self.dedent()

    def visit_With(self, node):
        """Preserve with statement structure."""
        items = []
        for item in node.items:
            context = ast.unparse(item.context_expr) if hasattr(item, 'context_expr') else "..."
            var = f" as {ast.unparse(item.optional_vars)}" if item.optional_vars else ""
            items.append(context + var)
            
        self.add_line(f"with {', '.join(items)}:")
        self.indent()
        for item in node.body:
            self.visit(item)
        self.dedent()

    def visit_Assign(self, node):
        """Keep assignment statements with original variable names."""
        targets = ", ".join(ast.unparse(t) for t in node.targets)
        self.add_line(f"{targets} = ...")

    def visit_AugAssign(self, node):
        """Keep augmented assignment with original variable names."""
        target = ast.unparse(node.target)
        op = self._get_operator(node.op)
        self.add_line(f"{target} {op}= ...")

    def visit_Return(self, node):
        """Keep return statements with their full content."""
        if node.value:
            try:
                # Preserve the complete return expression
                return_expr = ast.unparse(node.value)
                self.add_line(f"return {return_expr}")
            except Exception as e:
                print(f"[WARN] Failed to unparse return expression: {e}")
                self.add_line("return ...")
        else:
            self.add_line("return")

    def visit_Expr(self, node):
        """Handle expression statements - preserve call structure."""
        if isinstance(node.value, ast.Call):
            # Keep function calls with original names
            try:
                call_expr = ast.unparse(node.value)
                # Simplify complex expressions by replacing arguments with ...
                func_name = ast.unparse(node.value.func)
                self.add_line(f"{func_name}(...)")
            except:
                self.add_line("...(...)")
        elif isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            # Skip string literals (docstrings)
            pass
        else:
            self.add_line("...")

    def visit_Pass(self, node):
        self.add_line("pass")

    def visit_Break(self, node):
        self.add_line("break")

    def visit_Continue(self, node):
        self.add_line("continue")

    def _get_operator(self, op_node) -> str:
        """Get operator symbol."""
        operators = {
            ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/", 
            ast.Mod: "%", ast.Pow: "**", ast.LShift: "<<", ast.RShift: ">>",
            ast.BitOr: "|", ast.BitXor: "^", ast.BitAnd: "&", ast.FloorDiv: "//",
            ast.MatMult: "@", ast.And: "and", ast.Or: "or", ast.Not: "not",
            ast.UAdd: "+", ast.USub: "-", ast.Not: "not",
            ast.Eq: "==", ast.NotEq: "!=", ast.Lt: "<", ast.LtE: "<=",
            ast.Gt: ">", ast.GtE: ">=", ast.Is: "is", ast.IsNot: "is not",
            ast.In: "in", ast.NotIn: "not in"
        }
        return operators.get(type(op_node), "?")

    def generate_skeleton(self, code: str) -> str:
        """Generate structured skeleton from code."""
        self.reset_state()
        
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return self._fallback_skeleton(code)
            
        # Process imports first
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                self.visit(node)
                
        if self.imports:
            self.add_line()
            
        # Process other nodes
        for node in tree.body:
            if not isinstance(node, (ast.Import, ast.ImportFrom)):
                self.visit(node)
                
        # Build final output with imports at top
        header = []
        if self.imports:
            header.extend(sorted(self.imports))
            
        return "\n".join(header + self.lines).strip()

    def _fallback_skeleton(self, code: str) -> str:
        """Fallback using regex to extract structure while keeping all names and return statements."""
        lines = []
        imports = set()
        functions = []
        classes = []
        control_structures = []
        return_statements = []
        
        for line in code.splitlines():
            stripped = line.strip()
            
            # Capture imports
            if re.match(r'^(import|from)\s+', stripped):
                imports.add(stripped)
                continue
                
            # Capture function definitions - KEEP ALL NAMES
            if re.match(r'^def\s+[a-zA-Z_]\w*\s*\(', stripped):
                # Extract function signature with original name
                match = re.match(r'(def\s+[a-zA-Z_]\w*\s*\([^)]*\))', stripped)
                if match:
                    functions.append(match.group(1) + ":")
                else:
                    functions.append(stripped + ":" if not stripped.endswith(':') else stripped)
                continue
                
            # Capture class definitions - KEEP ALL NAMES
            if re.match(r'^class\s+[a-zA-Z_]\w*', stripped):
                classes.append(stripped + ":" if not stripped.endswith(':') else stripped)
                continue
                
            # Capture return statements - KEEP FULL CONTENT
            if re.match(r'^return\s+', stripped):
                return_statements.append(stripped)
                continue
                
            # Capture control structures
            if any(stripped.startswith(keyword) for keyword in 
                  ['for ', 'while ', 'if ', 'elif ', 'else:', 'try:', 'except', 'finally:', 'with ']):
                control_structures.append(stripped)
                
        # Build output
        if imports:
            lines.extend(sorted(imports))
            lines.append("")
            
        for class_def in classes:
            lines.append(class_def)
            lines.append("    ...")
            lines.append("")
            
        for func_def in functions:
            lines.append(func_def)
            # Add placeholder for function body
            lines.append("    ...")
            # Add return statements if any
            for ret_stmt in return_statements:
                if "return" in ret_stmt:
                    lines.append(f"    {ret_stmt}")
            lines.append("")
            
        for control in control_structures:
            lines.append(control)
            if control.endswith(':'):
                lines.append("    ...")
            lines.append("")
            
        return "\n".join(lines).strip()


def extract_code_blocks(text: str) -> str:
    """Extract code from text, preferring code blocks."""
    if not isinstance(text, str) or not text.strip():
        return ""
        
    # Try code fences first
    code_match = re.search(r"```(?:\w+)?\s*\n(.*?)```", text, flags=re.DOTALL)
    if code_match:
        return code_match.group(1).strip()
        
    # If no code fence, look for indented code blocks
    lines = text.splitlines()
    code_lines = []
    in_code = False
    base_indent = 0
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        # Look for code starters
        if (stripped.startswith('def ') or stripped.startswith('class ') or 
            stripped.startswith('import ') or stripped.startswith('from ') or
            (stripped and i > 0 and line[:4].isspace() and lines[i-1].strip())):
            
            if not in_code:
                in_code = True
                # Set base indent for this code block
                base_indent = len(line) - len(line.lstrip())
                
            code_lines.append(line)
        elif in_code and stripped and (len(line) - len(line.lstrip())) >= base_indent:
            code_lines.append(line)
        elif in_code and not stripped:
            code_lines.append(line)
        else:
            if in_code and len(code_lines) > 10:  # Don't mix multiple code blocks
                break
            in_code = False
            
    result = "\n".join(code_lines).strip()
    return result if result else text.strip()


def read_question_file(task_dir: str) -> str:
    """Read question.txt file from task directory."""
    question_file = os.path.join(task_dir, "question.txt")
    if os.path.exists(question_file):
        try:
            with open(question_file, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception as e:
            print(f"[WARN] Failed to read question.txt for {task_dir}: {e}", file=sys.stderr)
    return ""


def read_solutions_file(task_dir: str) -> List[str]:
    """Read solutions.json file from task directory."""
    solutions_file = os.path.join(task_dir, "solutions.json")
    if os.path.exists(solutions_file):
        try:
            with open(solutions_file, "r", encoding="utf-8") as f:
                solutions_data = json.load(f)
                if isinstance(solutions_data, list):
                    return solutions_data
                else:
                    return [str(solutions_data)]
        except Exception as e:
            print(f"[WARN] Failed to read solutions.json for {task_dir}: {e}", file=sys.stderr)
    return []


def copy_original_files(task_dir: str, output_dir: str, task_id: str):
    """Copy original solutions.json and question.txt to output directory."""
    # Create task subdirectory in output
    task_output_dir = os.path.join(output_dir, task_id)
    os.makedirs(task_output_dir, exist_ok=True)
    
    # Copy solutions.json
    solutions_src = os.path.join(task_dir, "solutions.json")
    solutions_dst = os.path.join(task_output_dir, "solutions.json")
    if os.path.exists(solutions_src):
        try:
            shutil.copy2(solutions_src, solutions_dst)
        except Exception as e:
            print(f"[WARN] Failed to copy solutions.json for {task_id}: {e}", file=sys.stderr)
    
    # Copy question.txt
    question_src = os.path.join(task_dir, "question.txt")
    question_dst = os.path.join(task_output_dir, "question.txt")
    if os.path.exists(question_src):
        try:
            shutil.copy2(question_src, question_dst)
        except Exception as e:
            print(f"[WARN] Failed to copy question.txt for {task_id}: {e}", file=sys.stderr)


def process_directory(input_dir: str, output_path: str, copy_files: bool = True):
    """Process all solutions in directory and generate structured skeletons.
    
    Args:
        input_dir: Input directory containing task subdirectories
        output_path: Output JSONL file path
        copy_files: Whether to copy original solutions.json and question.txt files
    """
    generator = StructuredSkeletonGenerator()
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Find all task directories (subdirectories containing solutions.json)
    task_dirs = []
    for root, dirs, files in os.walk(input_dir):
        if "solutions.json" in files:
            task_dirs.append(root)
    
    print(f"[INFO] Found {len(task_dirs)} task directories")
    
    # Group by task
    task_data = defaultdict(lambda: {
        "solution_skeletons": [],  
        "original_solutions": [],
        "question_text": ""
    })
    
    for task_dir in task_dirs:
        try:
            task_id = os.path.basename(task_dir)
            
            # Read question.txt and solutions.json
            question_text = read_question_file(task_dir)
            original_solutions = read_solutions_file(task_dir)
            
            # Update task data
            task_data[task_id]["question_text"] = question_text
            task_data[task_id]["original_solutions"] = original_solutions
            
            # Copy original files if requested
            if copy_files:
                copy_original_files(task_dir, output_dir, task_id)
            
            # Process each solution
            for solution in original_solutions:
                solution_text = solution if isinstance(solution, str) else ""
                code = extract_code_blocks(solution_text)
                
                if code:
                    skeleton = generator.generate_skeleton(code)
                    task_data[task_id]["solution_skeletons"].append(skeleton)  
                else:
                    task_data[task_id]["solution_skeletons"].append("# No code extracted") 
                        
        except Exception as e:
            print(f"[WARN] Failed to process {task_dir}: {e}")
            continue
            
    # Write output JSONL
    with open(output_path, "w", encoding="utf-8") as f:
        for task_id, data in task_data.items():
            output = {
                "task_id": task_id,
                "question_text": data["question_text"],
                "original_solutions": data["original_solutions"],
                "solution_skeletons": data["solution_skeletons"],  
                "skeleton_count": len(data["solution_skeletons"])  
            }
            f.write(json.dumps(output, ensure_ascii=False) + "\n")
            
    total_skeletons = sum(len(data["solution_skeletons"]) for data in task_data.values()) 
    print(f"[INFO] Generated {total_skeletons} solution skeletons for {len(task_data)} tasks")  
    
    if copy_files:
        print(f"[INFO] Copied original solutions.json and question.txt files to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate code skeletons preserving all names and return statements")
    parser.add_argument("--input", "-i", required=True, help="Input directory containing task directories")
    parser.add_argument("--output", "-o", default="structuct_skeletons.jsonl",  
                       help="Output JSONL file path")
    parser.add_argument("--no-copy", action="store_true", 
                       help="Don't copy original solutions.json and question.txt files")
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.input):
        print(f"[ERROR] Input directory not found: {args.input}")
        sys.exit(1)
        
    process_directory(args.input, args.output, not args.no_copy)


if __name__ == "__main__":
    main()