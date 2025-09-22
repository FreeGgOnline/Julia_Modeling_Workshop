### A Pluto.jl notebook ###
# v0.20.18

using Markdown
using InteractiveUtils

# ╔═╡ 1a2b3c4d-5e6f-7890-abcd-ef1234567890
begin
    using Pkg
    Pkg.activate(temp=true)
    Pkg.add("PlutoUI")
    using PlutoUI
end

# ╔═╡ 2b3c4d5e-6f78-90ab-cdef-234567890123
md"""
# Getting Started with Julia: Installation, Tooling, and Community

*An introduction to the Julia ecosystem for scientific computing*

This notebook provides a comprehensive guide to getting started with Julia, including:
- Installation and setup
- Understanding Julia's mental model
- Development tools (VS Code, Pluto, Jupyter)
- Package management and Git workflows
- Community resources and documentation
- Best practices for scientific computing

Based on materials from the [UCI Data Science Initiative](https://github.com/UCIDataScienceInitiative/IntroToJulia) and modern Julia workflows.
"""

# ╔═╡ 3c4d5e6f-7890-abcd-ef12-345678901234
md"""
## 📥 Installing Julia

There are two recommended ways to install Julia:

### Option 1: Direct Download (Simple)
Visit [julialang.org/downloads](https://julialang.org/downloads/) and download the binary for your operating system.

### Option 2: Juliaup (Recommended for Long-term Use)
Juliaup makes it easy to manage and update Julia versions:

**Windows:** Install from the Microsoft Store

**Mac/Linux:** Run in terminal:
```bash
curl -fsSL https://install.julialang.org | sh
```

### Essential Juliaup Commands
- `juliaup help` - Show all commands
- `juliaup update` - Update Julia to latest version
- `juliaup default release` - Use latest stable release (recommended)
- `juliaup default lts` - Use long-term support version

After installation, type `julia` in your terminal to start the REPL!
"""

# ╔═╡ 4d5e6f78-90ab-cdef-2345-678901234567
md"""
## 🧠 Julia Mental Model: Talking to a Scientist

Understanding Julia's philosophy helps you write better code:

### Julia vs Other Languages

**Python/R/MATLAB:** Like talking to a politician
- They try to give you what you want
- May hide complexity behind the scenes
- Prioritize ease over speed

**C/Fortran:** Like talking to a philosopher
- Demand extreme specificity
- Require deep understanding of details
- Fast but verbose

**Julia:** Like talking to a scientist
- Surface simplicity with specific underlying details
- Context determines meaning (multiple dispatch)
- Nothing is hidden - you can inspect everything
- Expects precision but rewards with performance

### Key Principles
1. **Write generic code, get specific performance** - The compiler specializes your code
2. **Two-language problem solved** - Prototype and production in the same language
3. **Composability** - Packages work together naturally through shared abstractions
4. **Speed without sacrifice** - Fast code that's still readable
"""

# ╔═╡ 5e6f7890-abcd-ef12-3456-789012345678
md"""
## 🛠️ Development Tools

### VS Code (Recommended IDE)
The Julia VS Code extension provides a full-featured development environment:

1. **Install VS Code:** Download from [code.visualstudio.com](https://code.visualstudio.com/)
2. **Install Julia Extension:** Search for "Julia" in extensions (Ctrl/Cmd+Shift+X)
3. **Features:**
   - IntelliSense and autocomplete
   - Integrated REPL and debugger
   - Plot viewer and workspace explorer
   - Git integration
   - Linting and formatting

### Pluto Notebooks (Interactive Documents)
Perfect for reproducible research and teaching:

```julia
using Pkg
Pkg.add("Pluto")
using Pluto
Pluto.run()
```

**Why Pluto?**
- **Reproducible:** Package environments are built into the file
- **Reactive:** Changes propagate automatically
- **Simple:** No hidden state or execution order issues

### Jupyter Notebooks (Data Science)
For data analysis and visualization:

```julia
using Pkg
Pkg.add("IJulia")
using IJulia
notebook()
```

**When to use what:**
- **VS Code:** Package development, large projects
- **Pluto:** Teaching, reproducible research, reactive programming
- **Jupyter:** Data exploration, machine learning workflows
"""

# ╔═╡ 6f789012-3456-789a-bcde-f12345678901
md"""
## 📦 Package Management

Julia's package manager is built into the REPL. Press `]` to enter package mode:

### Essential Package Commands
- `help` - Show all commands
- `st` or `status` - Show installed packages
- `add PackageName` - Install a package
- `rm PackageName` - Remove a package
- `update` - Update all packages
- `activate .` - Activate environment in current directory
- `instantiate` - Install all packages from Project.toml

### Creating Environments
Environments isolate package dependencies per project:

```julia
# In package mode (])
activate MyProject
add DataFrames Plots
```

This creates `Project.toml` and `Manifest.toml` files that exactly specify your dependencies.

### From Code
```julia
using Pkg
Pkg.add("PackageName")
Pkg.activate("path/to/environment")
```
"""

# ╔═╡ 7890abcd-ef12-3456-789a-bcdef1234567
md"""
## 🌐 Git and Package Development

### Using Git with Julia Projects

1. **Initialize repository:**
```bash
git init
git add Project.toml Manifest.toml src/
git commit -m "Initial commit"
```

2. **Clone and instantiate:**
```bash
git clone https://github.com/username/MyPackage.jl
cd MyPackage.jl
julia --project -e 'using Pkg; Pkg.instantiate()'
```

### Developing Packages
```julia
# In package mode
dev path/to/MyPackage    # Use local version
dev MyPackage            # Clone from registry
```
"""

# ╔═╡ 890abcde-f123-4567-89ab-cdef12345678
md"""
## 👥 Community Resources

### Primary Communication Channels

📱 **[Julia Slack](https://julialang.org/slack/)** (Most Active)
- Real-time chat with core developers
- Channels for specific topics (#helpdesk, #machine-learning, #sciml)
- Friendly and responsive community

💬 **[Discourse Forum](https://discourse.julialang.org/)**
- In-depth discussions and questions
- Package announcements
- Performance help and code reviews

### Additional Resources

💬 **[Discord](https://discord.gg/julia)** - Community chat and discussions

📧 **[Zulip](https://julialang.zulipchat.com/)** - Threaded conversations

🐙 **[GitHub](https://github.com/JuliaLang/julia)** - Issues and development

📚 **[JuliaHub](https://juliahub.com/)** - Package search and documentation

### Getting Help

1. **Read the error message carefully** - Julia's errors are informative
2. **Search Discourse** - Your question may be answered
3. **Ask on Slack #helpdesk** - For quick questions
4. **Post on Discourse** - For complex issues needing code examples
5. **Check package issues** - For package-specific problems

### Community Guidelines
- Be respectful and patient
- Provide minimal working examples (MWE)
- Share what you've tried
- Help others when you can!
"""

# ╔═╡ 9abcdef0-1234-5678-9abc-def012345678
md"""
## 📚 Documentation and Learning Resources

### Official Documentation
- **[Julia Documentation](https://docs.julialang.org/)** - Comprehensive manual
- **[Julia by Example](https://juliabyexample.helpmanual.io/)** - Learn through code
- **[JuliaAcademy](https://juliaacademy.com/)** - Free courses

### Essential Reading
1. **[Noteworthy Differences](https://docs.julialang.org/en/v1/manual/noteworthy-differences/)** - Coming from other languages
2. **[Performance Tips](https://docs.julialang.org/en/v1/manual/performance-tips/)** - Write fast code
3. **[Style Guide](https://docs.julialang.org/en/v1/manual/style-guide/)** - Julia conventions
4. **[MATLAB-Python-Julia Cheatsheet](http://cheatsheets.quantecon.org/)** - Syntax comparison

### Modern Julia Workflows
- **[Modern Julia Workflows](https://modernjuliaworkflows.github.io/)** - Best practices and tools
- **[SciML Tutorials](https://tutorials.sciml.ai/)** - Scientific machine learning
- **[Julia Data Science](https://juliadatascience.io/)** - Data science with Julia

### Getting Documentation in the REPL
```julia
?println  # Shows documentation for println
```

Press `?` to enter help mode, then type any function or type name!
"""

# ╔═╡ 3456789a-bcde-f012-3456-789abcdef012
md"""
---
*This notebook is part of the Julia Modeling Workshop. For more materials, visit the [workshop repository](https://github.com/SciML/Julia_Modeling_Workshop).*
"""

# ╔═╡ Cell order:
# ╟─1a2b3c4d-5e6f-7890-abcd-ef1234567890
# ╟─2b3c4d5e-6f78-90ab-cdef-234567890123
# ╟─3c4d5e6f-7890-abcd-ef12-345678901234
# ╟─4d5e6f78-90ab-cdef-2345-678901234567
# ╟─5e6f7890-abcd-ef12-3456-789012345678
# ╟─6f789012-3456-789a-bcde-f12345678901
# ╟─7890abcd-ef12-3456-789a-bcdef1234567
# ╟─890abcde-f123-4567-89ab-cdef12345678
# ╟─9abcdef0-1234-5678-9abc-def012345678
# ╠═3456789a-bcde-f012-3456-789abcdef012
