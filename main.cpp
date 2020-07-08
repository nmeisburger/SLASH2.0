#include <algorithm>
#include <iostream>
#include <set>

using namespace std;

int main() {
    set<pair<int, string>> myset;

    myset.insert({1, "hello"});
    myset.insert({2, "hello"});
    myset.insert({1, "heloo"});
    myset.insert({1, "hello"});
    myset.insert({1, "hello"});
    myset.insert({1, "hello"});
    myset.insert({1, "hello"});

    for (const auto &item : myset) {
        cout << item.first << ", " << item.second << endl;
    }

    cout << "Size: " << myset.size();

    return 0;
}