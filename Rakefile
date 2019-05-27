$build_dir = "build"

directory $build_dir

desc "clean all build artifacts"
task :clean do
    FileUtils.rm_rf $build_dir
end

desc "run conan to install / generate dependencies"
task :conan => "build" do
    Dir.chdir "build"
    sh "conan install .."
    Dir.chdir ".."
end

desc "run cmake to produce platform-specific build files"
task :cmake => :conan do
    Dir.chdir "build"

    cmake_cmd = "cmake "

    # Windows support
    # cmake_cmd += "-DCMAKE_BUILD_TYPE=Debug "
    # cmake_cmd += "-G \"Visual Studio 15 2017 Win64\" " \
    #     if Rake::Win32::windows?
    cmake_cmd += "../src "

    sh cmake_cmd

    Dir.chdir ".."
end

desc "build binaries"
task :bin => :cmake do    
    Dir.chdir "build"

    make_cmd = "make -j8"

    # Windows support
    # make_cmd =
    #         "msbuild /m #{$project_name}.sln " +
    #         "/p:Configuration=Release " +
    #         "/p:Platform=\"x64\" " +
    #         "" if Rake::Win32::windows?

    sh make_cmd

    Dir.chdir ".."
end

desc "run the test executable"
task :test => :bin do
    sh "build/bin/test-nn"
end
